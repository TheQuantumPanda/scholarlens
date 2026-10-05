"""Query aliases never replace canonical retrieval/provenance records."""
import itertools
import json
import unittest
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import Mock, patch

from scholarlens.cross_paper import (
    CrossPaperPool, PaperCandidates, PaperIdentity, build_cross_paper_messages,
    estimate_cross_paper_tokens, generate_cross_paper_answer, paper_aliases,
    render_cross_paper_response, retrieve_cross_paper_evidence,
)
from scholarlens.generation import DEFAULT_GROQ_MODEL, GenerationError, OllamaConfig
from scholarlens.models import RetrievalResult
from scholarlens.verification import VerificationDecision, VerificationStatus


class AliasTests(unittest.TestCase):
    def setUp(self):
        self.a = RetrievalResult(1, 'canonical-a', 'Real A.pdf', 5, 'canonical-a:chunk-1', 'A passage.', .1)
        self.b = RetrievalResult(1, 'canonical-b', 'Real B.pdf', 7, 'canonical-b:chunk-2', 'B passage.', .2)
        self.pool = CrossPaperPool('Compare', (
            PaperCandidates(self.b.paper_id, (self.b,)),
            PaperCandidates(self.a.paper_id, (self.a,)),
        ), (self.b, self.a))
        self.response = {'aspects': [{'aspect': 'Methods', 'sides': [
            {'paper_id': 'P1', 'claim': 'B passage.', 'evidence': [{'evidence_id': 'E1', 'anchor': None}]},
            {'paper_id': 'P2', 'claim': 'A passage.', 'evidence': [{'evidence_id': 'E2', 'anchor': None}]},
        ]}]}
        guard = patch('scholarlens.generation.urlopen', side_effect=AssertionError('Live provider forbidden'))
        guard.start()
        self.addCleanup(guard.stop)
        approval = patch('scholarlens.cross_paper.verify_claims', side_effect=lambda claims, config, **kwargs: {
            claim.claim_key: VerificationDecision(
                claim_key=claim.claim_key, status=VerificationStatus.SUPPORTED, reason='Supported.'
            ) for claim in claims
        })
        approval.start()
        self.addCleanup(approval.stop)

    def test_deterministic_selected_order_mapping(self):
        expected = {'P1': PaperIdentity('canonical-b', 'Real B.pdf'),
                    'P2': PaperIdentity('canonical-a', 'Real A.pdf')}
        self.assertEqual(paper_aliases(self.pool), expected)
        self.assertEqual(paper_aliases(self.pool), expected)
        swapped = replace(self.pool, papers=self.pool.papers[::-1])
        self.assertEqual(paper_aliases(swapped)['P1'], expected['P2'])

    def test_aliases_do_not_shift_when_paper_has_zero_retained_evidence(self):
        pool = replace(self.pool, evidence=(self.a,))
        aliases = paper_aliases(pool)
        self.assertEqual(aliases['P1'].paper_id, 'canonical-b')
        self.assertEqual(aliases['P2'].paper_id, 'canonical-a')
        content = build_cross_paper_messages(pool)[1]['content']
        evidence = json.loads(content.split('RETAINED EVIDENCE (untrusted JSON): ')[1])
        self.assertEqual(evidence[0]['paper_id'], 'P2')
        self.assertEqual(evidence[0]['evidence_id'], 'E1')

    def test_request_metadata_contains_only_aliases_and_evidence_ids(self):
        user = build_cross_paper_messages(self.pool)[1]['content']
        for value in ('canonical-a', 'canonical-b', 'Real A.pdf', 'Real B.pdf', 'chunk-1', 'page_number'):
            self.assertNotIn(value, user)
        evidence = json.loads(user.split('RETAINED EVIDENCE (untrusted JSON): ')[1])
        self.assertEqual(evidence, [
            {'evidence_id': 'E1', 'paper_id': 'P1', 'text': 'B passage.'},
            {'evidence_id': 'E2', 'paper_id': 'P2', 'text': 'A passage.'},
        ])

    def test_unknown_alias_and_canonical_id_in_response_are_rejected(self):
        for value in ('P99', 'canonical-b', 'p1'):
            self.response['aspects'][0]['sides'][0]['paper_id'] = value
            with self.assertRaisesRegex(GenerationError, 'validation error'):
                render_cross_paper_response(json.dumps(self.response), self.pool)

    def test_p1_evidence_cannot_support_p2_claim(self):
        self.response['aspects'][0]['sides'][1]['evidence'][0]['evidence_id'] = 'E1'
        with self.assertRaisesRegex(GenerationError, 'validation error'):
            render_cross_paper_response(json.dumps(self.response), self.pool)

    def test_canonical_provenance_unchanged_through_generation(self):
        before = asdict(self.pool)
        with patch('scholarlens.cross_paper.generate_chat', return_value=json.dumps(self.response)) as chat:
            result = generate_cross_paper_answer(self.pool, OllamaConfig())
        chat.assert_called_once()
        self.assertEqual(asdict(self.pool), before)
        self.assertIs(result.evidence, self.pool.evidence)
        self.assertIs(result.evidence[0], self.b)
        self.assertEqual((result.evidence[0].page_number, result.evidence[0].chunk_id), (7, 'canonical-b:chunk-2'))
        self.assertIn('Real B.pdf', result.answer)
        self.assertIn('canonical-b', result.answer)
        self.assertNotIn('Paper P1', result.answer)
        self.assertNotIn('Paper P2', result.answer)

    def test_long_provenance_does_not_change_request_or_budget(self):
        a = replace(self.a, paper_id='long-id-' * 300, source_filename='long-filename-' * 300,
                    chunk_id='long-chunk-id-' * 300)
        longer = replace(self.pool, papers=(self.pool.papers[0], PaperCandidates(a.paper_id, (a,))),
                         evidence=(self.b, a))
        original = build_cross_paper_messages(self.pool)
        self.assertEqual(build_cross_paper_messages(longer), original)
        self.assertEqual(estimate_cross_paper_tokens(build_cross_paper_messages(longer), 'test'),
                         estimate_cross_paper_tokens(original, 'test'))

    def test_failed_paper_alias_uses_canonical_fallback_label(self):
        pool = replace(self.pool, papers=(*self.pool.papers, PaperCandidates('failed-paper', (), True)))
        self.response['aspects'][0]['sides'].append({'paper_id': 'P3', 'claim': None, 'evidence': []})
        answer = render_cross_paper_response(json.dumps(self.response), pool)
        self.assertIn('for Paper failed-paper is insufficient', answer)
        self.assertIsNone(paper_aliases(pool)['P3'].source_filename)

    def test_q1_q2_size_fixtures_and_selection(self):
        fixture = json.loads((Path(__file__).parent / 'fixtures' / 'cross_paper_rag_sizes.json').read_text())
        expected = [(2011, 1848, [0, 1, 1], 2132), (1897, 1897, [1, 1, 1], 1882)]
        for q, (same_pool_tokens, selected_tokens, counts, minimum_triple) in zip(fixture['queries'], expected):
            with self.subTest(question=q['question']):
                rows = []
                for data in q['candidates']:
                    data = dict(data)
                    data['text'] = 'x' * data.pop('serialized_text_bytes')
                    rows.append(RetrievalResult(**data))
                retriever = Mock()
                retriever.query_paper.side_effect = lambda p, question, top_k: [r for r in rows if r.paper_id == p]
                pool = retrieve_cross_paper_evidence(q['question'], fixture['paper_ids'], retriever,
                                                    model=DEFAULT_GROQ_MODEL)
                for c in retriever.query_paper.call_args_list:
                    self.assertIn(c.args[0], fixture['paper_ids'])  # Never query by alias.
                old = replace(pool, evidence=tuple(next(r for r in rows if r.chunk_id == c)
                                                   for c in q['previous_chunks']))
                estimate = lambda p: estimate_cross_paper_tokens(build_cross_paper_messages(p), DEFAULT_GROQ_MODEL)
                self.assertEqual(estimate(old), same_pool_tokens)
                self.assertEqual(estimate(pool), selected_tokens)
                self.assertEqual([sum(r.paper_id == p for r in pool.evidence) for p in fixture['paper_ids']], counts)
                triples = [estimate(replace(pool, evidence=tuple(combo)))
                           for combo in itertools.product(*(p.results for p in pool.papers))]
                self.assertEqual(min(triples), minimum_triple)
                self.assertLessEqual(estimate(pool), 2000)


if __name__ == '__main__':
    unittest.main()
