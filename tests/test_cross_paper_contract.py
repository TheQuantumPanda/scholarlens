"""Structural grounding checks for the Phase 5B response contract."""
import copy
import json
import math
import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

from scholarlens.analysis import AnalysisConfig, estimate_grouped_request_tokens
from scholarlens.budget import estimate_request_tokens
from scholarlens.cross_paper import (
    CrossPaperConfig, CrossPaperPool, CrossPaperResponse, PaperCandidates,
    INSUFFICIENT_COMPARISON, build_cross_paper_messages, estimate_cross_paper_tokens,
    generate_cross_paper_answer, render_cross_paper_response, retrieve_cross_paper_evidence,
)
from scholarlens.generation import GenerationError, GroqConfig, OllamaConfig, build_chat_payload
from scholarlens.models import RetrievalResult
from scholarlens.verification import VerificationDecision, VerificationStatus


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.evidence = (
            RetrievalResult(1, 'a', 'Owned A.pdf', 3, 'c1', 'Designed to use NLP. Simulated results.', .1),
            RetrievalResult(1, 'b', 'Owned B.pdf', 5, 'c2', 'Training accuracy increased.', .2),
            RetrievalResult(2, 'a', 'Owned A.pdf', 4, 'c3', 'Separate passage.', .3),
        )
        self.pool = CrossPaperPool('Compare methods', tuple(PaperCandidates(p, tuple(
            r for r in self.evidence if r.paper_id == p)) for p in ('b', 'a')), self.evidence)
        self.response = {'aspects': [{'aspect': 'Methods', 'sides': [
            {'paper_id': 'P2', 'claim': 'Designed to use NLP.', 'evidence': [
                {'evidence_id': 'E1', 'anchor': 'Designed to use NLP.'}]},
            {'paper_id': 'P1', 'claim': 'Training accuracy increased.', 'evidence': [
                {'evidence_id': 'E2', 'anchor': 'Training accuracy increased.'}]},
        ]}]}
        self.http = patch('scholarlens.generation.urlopen', side_effect=AssertionError('No live HTTP'))
        self.http.start()
        self.addCleanup(self.http.stop)
        approval = patch('scholarlens.cross_paper.verify_claims', side_effect=lambda claims, config, **kwargs: {
            claim.claim_key: VerificationDecision(
                claim_key=claim.claim_key, status=VerificationStatus.SUPPORTED, reason='Supported.'
            ) for claim in claims
        })
        approval.start()
        self.addCleanup(approval.stop)

    def render(self):
        return render_cross_paper_response(json.dumps(self.response), self.pool)

    def first(self):
        return self.response['aspects'][0]['sides'][0]

    def invalid(self):
        with self.assertRaisesRegex(GenerationError, 'structured-response validation error'):
            self.render()

    def test_valid_two_paper_response_and_selected_order(self):
        answer = self.render()
        self.assertEqual(answer, 'Methods\n\nPaper Owned B.pdf \\(b\\): Training accuracy increased. [E2]\n\n'
                         'Paper Owned A.pdf \\(a\\): Designed to use NLP. [E1]')

    def test_unknown_id(self):
        self.first()['evidence'][0]['evidence_id'] = 'E999'
        self.invalid()

    def test_wrong_paper(self):
        self.first()['evidence'][0] = {'evidence_id': 'E2', 'anchor': None}
        self.invalid()

    def test_duplicate_refs(self):
        self.first()['evidence'] *= 2
        self.invalid()

    def test_missing_selected_side(self):
        self.pool = replace(self.pool, papers=(*self.pool.papers, PaperCandidates('c', ())))
        self.invalid()

    def test_duplicate_paper(self):
        self.response['aspects'][0]['sides'][1]['paper_id'] = 'P2'
        self.invalid()

    def test_unselected_paper(self):
        self.first()['paper_id'] = 'foreign'
        self.invalid()

    def test_exact_anchor_success(self):
        self.first()['evidence'][0]['anchor'] = 'use NLP.'
        self.assertIn('[E1]', self.render())

    def test_invalid_anchors(self):
        for anchor in ('designed to use NLP.', 'Designed ... NLP.', 'Separate passage.',
                       'Training accuracy increased.'):
            with self.subTest(anchor=anchor):
                self.first()['evidence'][0]['anchor'] = anchor
                self.invalid()

    def test_altered_hyphenation_anchor_rejected(self):
        self.evidence = (replace(self.evidence[0], text='Deter- mining retrieval quality.'),
                         *self.evidence[1:])
        self.pool = replace(self.pool, evidence=self.evidence)
        self.first()['evidence'][0]['anchor'] = 'Determining'
        self.invalid()

    def test_null_anchor_allowed(self):
        self.first()['evidence'][0]['anchor'] = None
        self.assertIn('[E1]', self.render())

    def test_prompt_requires_exact_optional_anchors_and_null_for_uncertainty(self):
        prompt = build_cross_paper_messages(self.pool)[0]['content']
        for requirement in ('Anchors are optional', 'exact contiguous substring',
                            'preserving extraction artifacts exactly', 'hyphenation',
                            'spacing', 'punctuation', 'capitalization', 'anchor=null',
                            'Never reconstruct, normalize, dehyphenate, or paraphrase'):
            self.assertIn(requirement, prompt)

    def test_prompt_allows_related_asymmetric_cross_paper_evidence(self):
        prompt = ' '.join(build_cross_paper_messages(self.pool)[0]['content'].split())
        for requirement in ('different terminology', 'need not directly agree or disagree',
                            'Evidence need not be symmetric',
                            'independently provide evidence',
                            "Each non-null claim must answer the question from that paper's own",
                            'do not infer a relationship the evidence does not support',
                            'fewer than two papers contain evidence that materially answers'):
            self.assertIn(requirement, prompt)

    def test_blank_anchor_rejected(self):
        for anchor in ('', ' \n '):
            self.first()['evidence'][0]['anchor'] = anchor
            self.invalid()

    def test_claim_without_evidence(self):
        self.first()['evidence'] = []
        self.invalid()

    def test_null_claim_with_evidence(self):
        self.first()['claim'] = None
        self.invalid()

    def test_blank_claim_and_aspect_rejected(self):
        original = copy.deepcopy(self.response)
        self.first()['claim'] = '  '
        self.invalid()
        self.response = original
        self.response['aspects'][0]['aspect'] = '  '
        self.invalid()

    def test_one_qualifying_and_one_one_sided_aspect(self):
        partial = copy.deepcopy(self.response['aspects'][0])
        partial['aspect'] = 'Evaluation'
        partial['sides'][0].update(claim=None, evidence=[])
        self.response['aspects'].append(partial)
        answer = self.render()
        self.assertIn('Paper Owned B.pdf \\(b\\): Training accuracy increased. [E2]', answer)
        self.assertIn('The supplied evidence for Paper Owned A.pdf \\(a\\) is insufficient to establish this aspect.', answer)

    def test_no_qualifying_aspect_exact_fallback(self):
        self.first().update(claim=None, evidence=[])
        self.assertEqual(self.render(), INSUFFICIENT_COMPARISON)
        self.response['aspects'] = []
        self.assertEqual(self.render(), INSUFFICIENT_COMPARISON)

    def test_extra_fields_at_every_level_rejected(self):
        original = copy.deepcopy(self.response)
        for level in ('root', 'aspect', 'side', 'ref'):
            self.response = copy.deepcopy(original)
            targets = {'root': self.response, 'aspect': self.response['aspects'][0],
                       'side': self.first(), 'ref': self.first()['evidence'][0]}
            targets[level]['summary'] = 'Invented summary'
            self.invalid()

    def test_strict_types_and_limits(self):
        original = copy.deepcopy(self.response)
        self.first()['claim'] = 123
        self.invalid()
        self.response = copy.deepcopy(original)
        self.response['aspects'] *= 4
        self.invalid()
        self.response = copy.deepcopy(original)
        self.first()['evidence'] *= 7
        self.invalid()
        self.response = copy.deepcopy(original)
        self.response['aspects'][0]['sides'] *= 3
        self.invalid()

    def test_invalid_output_not_rendered_or_repaired(self):
        for raw in ('PRIVATE RAW OUTPUT', '{"aspects": [], "summary": "PRIVATE RAW OUTPUT"}'):
            with patch('scholarlens.cross_paper.generate_chat', return_value=raw) as chat:
                with self.assertRaises(GenerationError) as caught:
                    generate_cross_paper_answer(self.pool, OllamaConfig())
                self.assertNotIn('PRIVATE RAW', str(caught.exception))
                chat.assert_called_once()

    def test_provider_neutral_schema_and_snapshot(self):
        for config in (OllamaConfig(), GroqConfig(api_key='fake')):
            with patch('scholarlens.cross_paper.generate_chat', return_value=json.dumps(self.response)) as chat:
                answer = generate_cross_paper_answer(self.pool, config)
            self.assertEqual(answer.evidence, self.evidence)
            self.assertEqual(chat.call_args.kwargs['response_schema'], CrossPaperResponse.model_json_schema())
            self.assertIn('Paper Owned B.pdf \\(b\\):', answer.answer)
            self.assertIn('Owned A.pdf', answer.answer)

    def test_inline_citations_cannot_bypass_refs(self):
        self.first()['claim'] = 'Invented citation [E999]'
        self.invalid()

    def test_markdown_in_claim_cannot_create_links(self):
        self.first()['claim'] = '[Fake provenance](https://example.invalid)'
        self.assertIn(r'\[Fake provenance\]\(https://example.invalid\)', self.render())

    def test_unsupported_paraphrase_can_pass_structural_validation(self):
        # Explicit limitation: exact anchors and paper ownership are not entailment.
        self.first()['claim'] = 'The system implements a transformer trained on a million records.'
        self.assertIn('implements a transformer', self.render())

    def test_schema_charged_at_message_rate_with_full_framing(self):
        messages = build_cross_paper_messages(self.pool)
        schema = CrossPaperResponse.model_json_schema()
        sizes = [math.ceil(len(json.dumps(build_chat_payload(messages, 'test', provider=p,
                 response_schema=schema)).encode()) / 6 * 1.6) for p in ('groq', 'ollama')]
        self.assertEqual(estimate_cross_paper_tokens(messages, 'test'), max(sizes))
        self.assertGreater(max(sizes), estimate_request_tokens(messages, 'test', schema=schema))

    def test_schema_adaptation_counted(self):
        schema = {'type': 'object', 'properties': {'x': {'const': 'x', 'pattern': 'x'}}}
        with patch.object(CrossPaperResponse, 'model_json_schema', return_value=schema):
            payload = build_chat_payload([], 'test', provider='groq', response_schema=schema)
            adapted = payload['response_format']['json_schema']['schema']
            self.assertEqual(adapted['properties']['x'], {'enum': ['x']})
            expected = max(math.ceil(len(json.dumps(build_chat_payload([], 'test', provider=p,
                response_schema=schema)).encode()) / 6 * 1.6) for p in ('groq', 'ollama'))
            self.assertEqual(estimate_cross_paper_tokens([], 'test'), expected)

    def test_historical_diagnostic_accounting_without_private_pdf_fixture(self):
        # Byte-sized stand-in for the historical E1-E4 request (old estimate 1850).
        # No copyrighted/local paper passages are stored in the test suite.
        messages = [{'role': 'user', 'content': ''}]
        base = len(json.dumps({'model': 'qwen/qwen3.8-27b', 'messages': messages,
                               'stream': False}, ensure_ascii=False).encode())
        messages[0]['content'] = 'x' * (8878 - base)
        self.assertEqual(estimate_request_tokens(messages, 'qwen/qwen3.8-27b'), 1850)
        self.assertEqual(estimate_request_tokens(messages, 'qwen/qwen3.8-27b',
                                               estimate_safety_factor=1.6), 2368)
        self.assertGreater(estimate_cross_paper_tokens(messages, 'qwen/qwen3.8-27b'), 2000)

    def test_selection_and_preflight_share_exact_accounting(self):
        retriever = Mock()
        retriever.query_paper.side_effect = lambda p, q, top_k: [e for e in self.evidence if e.paper_id == p]
        pool = retrieve_cross_paper_evidence('Compare methods', ('b', 'a'), retriever, model='test')
        limit = estimate_cross_paper_tokens(build_cross_paper_messages(pool), 'test')
        config = OllamaConfig(model='test')
        with patch('scholarlens.cross_paper.generate_chat', return_value='{"aspects": []}') as chat:
            generate_cross_paper_answer(pool, config, budget=CrossPaperConfig(safe_prompt_tokens=limit))
            chat.assert_called_once()
        with patch('scholarlens.cross_paper.generate_chat') as chat:
            with self.assertRaisesRegex(GenerationError, 'prompt budget'):
                generate_cross_paper_answer(pool, config, budget=CrossPaperConfig(safe_prompt_tokens=limit-1))
            chat.assert_not_called()

    def test_fixed_oversized_pool_rejected_without_provider_call(self):
        large = replace(self.pool, evidence=tuple(replace(e, text=e.text * 500) for e in self.evidence))
        with patch('scholarlens.cross_paper.generate_chat') as chat:
            with self.assertRaisesRegex(GenerationError, 'prompt budget'):
                generate_cross_paper_answer(large, GroqConfig(api_key='fake'))
            chat.assert_not_called()

    def test_individual_analysis_accounting_unchanged(self):
        messages, schema = [{'role': 'user', 'content': 'Test'}], {'type': 'object'}
        config = AnalysisConfig()
        expected = estimate_request_tokens(messages, 'test', schema=schema,
            estimated_bytes_per_token=6, schema_bytes_per_token=32, estimate_safety_factor=1.25)
        self.assertEqual(estimate_grouped_request_tokens(messages, schema, 'test', config), expected)


if __name__ == '__main__':
    unittest.main()
