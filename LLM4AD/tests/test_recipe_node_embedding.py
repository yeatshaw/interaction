"""Offline regression checks for response-based recipe contexts."""

import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from example.tasks.mcts_recipe_runner import OpenAIEmbedding
from example.tasks import mcts_recipe_runner as runner
from llm4ad.base import TextFunctionProgramConverter, SampleTrimmer
from llm4ad.method.eoh.sampler import EoHSampler
from llm4ad.method.mcts_recipe.mcts_recipe import MCTSRecipe
from llm4ad.method.mcts_recipe.node_embedding import NodeEmbedding, NodeEmbeddingError
from llm4ad.method.mcts_recipe.refineevo_init import sample_initial_algorithm
from llm4ad.method.mcts_recipe.persistence import RecipeStore
from llm4ad.method.mcts_recipe.population import RecipePopulation
from llm4ad.method.recipe_selection import RecipeSelector


class Encoder:
    def __init__(self):
        self.calls = []

    def encode(self, texts):
        self.calls.append(list(texts))
        return [[float(text), float(text) * 2] for text in texts]


def individual(number):
    function = TextFunctionProgramConverter.text_to_function(
        f"def heuristic(x):\n    return x + {number}\n")
    function.score = float(number)
    function.algorithm = str(number)
    function._generation_raw_response = str(number)
    return function


class NodeEmbeddingTests(unittest.TestCase):
    def test_embedding_503_retries_same_response_without_regenerating(self):
        raw = "Thought: {algorithm}\n```python\ndef heuristic(x):\n    return x\n```"
        llm = SimpleNamespace(draw_sample=unittest.mock.Mock(return_value=raw))
        client = OpenAIEmbedding("key", "https://example.test/v1", "text-embedding-v4",
                                 dimensions=64, max_retries=2)
        sampler = EoHSampler(llm, "def heuristic(x):\n    return x\n",
                             response_encoder=NodeEmbedding(client, 64).encode_response)
        success = unittest.mock.MagicMock()
        success.__enter__.return_value.read.return_value = json.dumps({
            "data": [{"index": 0, "embedding": list(range(64))}]
        }).encode()
        error = urllib.error.HTTPError("https://example.test/v1/embeddings", 503,
                                       "Service Unavailable", {"Retry-After": "2"}, None)
        with patch("urllib.request.urlopen", side_effect=[error, success]) as request, \
                patch.object(runner.time, "sleep") as sleep:
            _, function = sampler.get_thought_and_function("operator prompt")
        llm.draw_sample.assert_called_once()
        self.assertEqual(request.call_count, 2)
        for call in request.call_args_list:
            self.assertEqual(json.loads(call.args[0].data)["input"], [raw])
        sleep.assert_called_once_with(2.0)
        self.assertEqual(function._node_embedding, list(range(64)))

    def test_embedding_retry_limit_and_permanent_errors(self):
        for code, attempts in ((503, 3), (401, 1)):
            with self.subTest(code=code):
                client = OpenAIEmbedding("key", "https://example.test/v1", "text-embedding-v4",
                                         max_retries=2)

                def fail(*args, **kwargs):
                    raise urllib.error.HTTPError("https://example.test/v1/embeddings",
                                                  code, "test failure", {}, None)

                with patch("urllib.request.urlopen", side_effect=fail) as request, \
                        patch.object(runner.time, "sleep") as sleep:
                    with self.assertRaises(urllib.error.HTTPError) as error:
                        client.encode("same response")
                self.assertEqual(error.exception.code, code)
                self.assertEqual(request.call_count, attempts)
                self.assertEqual(sleep.call_count, attempts - 1)

    def test_embedding_request_concurrency_limit(self):
        active = 0
        peak = 0
        lock = runner.threading.Lock()

        class Response:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def read(self):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                try:
                    runner.time.sleep(0.02)
                    return b'{"data":[{"index":0,"embedding":[1,2]}]}'
                finally:
                    with lock:
                        active -= 1

        client = OpenAIEmbedding("key", "https://example.test/v1", "text-embedding-v4",
                                 max_concurrent_requests=2)
        with patch("urllib.request.urlopen", side_effect=lambda *a, **k: Response()), \
                runner.concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
            results = list(executor.map(client.encode, ["response"] * 6))
        self.assertLessEqual(peak, 2)
        self.assertEqual(len(results), 6)

    def test_generation_embeds_full_response_before_parsing(self):
        response = "Thought: {full response}\n```python\ndef heuristic(x):\n    return x\n```"
        calls = []

        def encode(text):
            calls.append(text)
            return np.arange(64, dtype=float)

        encoder = NodeEmbedding(SimpleNamespace(encode=encode), 64)
        llm = SimpleNamespace(draw_sample=lambda _: response)
        sampler = EoHSampler(llm, "def heuristic(x):\n    return x\n",
                             response_encoder=encoder.encode_response)
        for sample in (sampler.get_thought_and_function,
                       sampler.get_recipe_thought_and_function):
            calls.clear()

            def parse(*args):
                self.assertEqual(calls, [response])
                return individual(1)

            with patch.object(SampleTrimmer, "sample_to_function", side_effect=parse):
                _, function = sample("operator prompt")
            record = RecipeStore._algorithm_record(function)
            self.assertNotIn("generation_raw_response", record)
            self.assertEqual(record["embedding"], list(range(64)))
            with tempfile.TemporaryDirectory() as directory:
                store = RecipeStore(directory, batch_size=1)
                store.add(1, None, 0, None, 0, RecipePopulation([function], 1))
                saved = store.load_node(1)["algorithms"][0]
                self.assertNotIn("generation_raw_response", saved)
                self.assertEqual(saved["embedding"], list(range(64)))
            # Saving/aggregating this population must not call the API again.
            encoder.encode_population(RecipePopulation([function], 1))
            self.assertEqual(calls, [response])
        calls.clear()
        _, function = sample_initial_algorithm(
            llm, "initial prompt", "def heuristic(x):\n    return x\n",
            response_encoder=encoder.encode_response)
        self.assertEqual(function._node_embedding, list(range(64)))
        self.assertEqual(calls, [response])

    def test_embedding_failure_is_not_a_parse_failure(self):
        encoder = NodeEmbedding(SimpleNamespace(encode=lambda _: [1]), 64)
        sampler = EoHSampler(SimpleNamespace(draw_sample=lambda _: "raw response"),
                             "def heuristic(x):\n    return x\n",
                             response_encoder=encoder.encode_response)
        with patch.object(SampleTrimmer, "sample_to_function") as parser:
            with self.assertRaises(NodeEmbeddingError):
                sampler.get_thought_and_function("operator prompt")
            parser.assert_not_called()

    def test_all_initialization_modes_embed_before_evaluation(self):
        response = "Thought: {initial algorithm}\n```python\ndef heuristic(x):\n    return x\n```"
        info = {"template_program": "def heuristic(x):\n    return x\n",
                "template_module": "unused"}
        for initialize in (runner._initial_population_eoh_recipe,
                           runner._initial_population_eoh,
                           runner._initial_population):
            with self.subTest(mode=initialize.__name__):
                encoded = []
                recorded = []

                def encode(text):
                    encoded.append(text)
                    return np.arange(64, dtype=float)

                def evaluate(program):
                    self.assertEqual(encoded, [response])
                    return 1.0, 0.01

                evaluator = SimpleNamespace(evaluate_program_record_time=evaluate)
                options = {"init_pop_size": 1} if initialize is runner._initial_population else {}
                with patch.object(runner, "SecureEvaluator", return_value=evaluator), \
                        patch.object(runner.EoHPrompt, "get_prompt_i1", return_value="init"), \
                        patch.object(runner, "_refineevo_initial_prompt", return_value="init"):
                    population, _ = initialize(
                        SimpleNamespace(draw_sample=lambda _: response), None, info,
                        pop_size=1, selection_num=1, num_samplers=1,
                        on_evaluated=lambda function: recorded.append(
                            RecipeStore._algorithm_record(function)),
                        response_encoder=NodeEmbedding(
                            SimpleNamespace(encode=encode), 64).encode_response,
                        **options)
                self.assertEqual(len(population), 1)
                self.assertEqual(len(recorded), 1)
                self.assertEqual(recorded[0]["embedding"], list(range(64)))
                self.assertEqual(encoded, [response])

    def test_raw_generation_response_survives_parsing(self):
        response = "Thought: {test}\n```python\ndef heuristic(x):\n    return x + 1\n```"
        sampler = EoHSampler(
            SimpleNamespace(draw_sample=lambda _: response),
            "def heuristic(x):\n    return x\n")
        for sample in (sampler.get_thought_and_function,
                       sampler.get_recipe_thought_and_function):
            _, function = sample("prompt")
            self.assertEqual(function._generation_raw_response, response)

    def test_independent_embedding_request_dimensions(self):
        requests = []

        def respond(request, timeout):
            requests.append(json.loads(request.data))
            response = unittest.mock.MagicMock()
            response.__enter__.return_value.read.return_value = json.dumps({
                "data": [{"index": 0, "embedding": [1, 2]}]
            }).encode()
            return response

        experience = OpenAIEmbedding("experience-key", "https://example.test/v1", "text-embedding-v4")
        node = OpenAIEmbedding("node-key", "https://example.test/v1", "text-embedding-v4", dimensions=2)
        with patch("urllib.request.urlopen", side_effect=respond):
            experience.encode("experience")
            node.encode("raw response")
        self.assertNotIn("dimensions", requests[0])
        self.assertEqual(requests[1]["dimensions"], 2)

    def test_embedding_shape_and_missing_response_fail(self):
        context = NodeEmbedding(Encoder(), 3)
        with self.assertRaises(ValueError):
            context.encode_population(RecipePopulation([individual(1)], 1))
        with self.assertRaises(ValueError):
            NodeEmbedding(Encoder(), 2).encode_population(
                RecipePopulation([SimpleNamespace()], 1))

    def test_selector_dimension_checkpoint_validation(self):
        selector = RecipeSelector(["a", "b"], context_dim=2)
        selector.update("a", [1, 2], 3)
        restored = RecipeSelector(["a", "b"], context_dim=2)
        restored.load_state_dict(selector.state_dict())
        np.testing.assert_allclose(restored.b["a"], [3, 3, 6])
        with self.assertRaises(ValueError):
            RecipeSelector(["a", "b"]).load_state_dict(selector.state_dict())

    def test_single_vote_top_k(self):
        selector = RecipeSelector(["a", "b", "c", "d", "e"], context_dim=1)
        winners = np.asarray([0, 0, 1, 1, 2])
        draws = [np.column_stack((10 * (winners == arm), np.zeros(5)))
                 for arm in range(5)]
        with patch("numpy.random.multivariate_normal", side_effect=draws) as sampler:
            selected, details = selector.select_population_top_k(
                [[0]] * 5, list(selector.recipe_ids), k=3)
        self.assertEqual(set(selected), {"a", "b", "c"})
        self.assertEqual(details["vote_counts"], {"a": 2, "b": 2, "c": 1, "d": 0, "e": 0})
        self.assertEqual(details["individual_votes"], ["a", "a", "b", "b", "c"])
        self.assertEqual(sum(details["vote_counts"].values()), 5)
        for call in sampler.call_args_list:
            self.assertEqual(call.kwargs["size"], 5)
        selected, details = selector.select_population_top_k([[0]] * 5, ["c"], k=3)
        self.assertEqual(selected, ["c"])
        self.assertEqual(details["vote_counts"], {"c": 5})

    def test_repeat_mode_keeps_nodes_expandable_without_root_starvation(self):
        from llm4ad.method.mcts_recipe.mcts import RecipeMCTS, RecipeNode
        tree = RecipeMCTS({"a": {}, "b": {}}, allow_recipe_repeats=True)
        root = RecipeNode(1, 0, visits=20)
        child = RecipeNode(2, 1, depth=1, parent=root)
        root.add_child(child)
        root.expanded_recipe_ids = {"a", "b"}
        tree.root = root
        self.assertIn(root, tree._expandable_nodes())
        self.assertTrue(np.isfinite(tree.uct(root)))
        tree.allow_recipe_repeats = False
        self.assertNotIn(root, tree._expandable_nodes())

    def test_repeat_setting_controls_expansion_and_disk_selection(self):
        for repeat in (False, True):
            with self.subTest(repeat=repeat), tempfile.TemporaryDirectory() as directory:
                encoder = Encoder()

                def expand(population, recipe, selection_num, target_size,
                           on_evaluated, reserve_sample_order):
                    child = individual(3)
                    child._recipe_sample_order = reserve_sample_order()
                    on_evaluated(child)
                    return [child]

                def build(allow_repeats):
                    return MCTSRecipe(
                        recipes={"a": {"comparison": True}, "b": {"comparison": True}},
                        pop_size=2, selection_num=1, max_depth=1, store_dir=directory,
                        expand_fn=expand, max_sample_count=100,
                        adaptive_config={"enabled": True, "recipes_per_expansion": 1,
                                         "allow_recipe_repeats": allow_repeats},
                        node_embedding_model=encoder,
                        node_embedding_config={"enabled": True, "dimensions": 2})

                method = build(repeat)
                initial = [individual(1), individual(2)]
                for function in initial:
                    function._recipe_sample_order = method.reserve_sample_order()
                root = method.initialize(initial)
                population, experiences = method._load_node_state(root.population_node_id)

                def choose_first(states, candidates, k):
                    return candidates[:k], {
                        "vote_counts": {arm: len(states) if index == 0 else 0
                                        for index, arm in enumerate(candidates)},
                        "individual_votes": [candidates[0]] * len(states),
                        "samples_per_individual": 1,
                    }

                created = []
                with patch.object(method.adaptive_router, "select_population_top_k",
                                  side_effect=choose_first):
                    for _ in range(2):
                        children = method.expand_node(root, population, experiences)
                        created.extend(children)
                        method._write_disk_tree_transition(root, children)
                        method.store.flush()
                        method._commit_pending_tree_state()
                self.assertEqual([node.incoming_recipe_id for node in created],
                                 ["a", "a"] if repeat else ["a", "b"])
                self.assertEqual(len({node.population_node_id for node in created}), 2)
                selected = method._select_disk_tree_node()
                if repeat:
                    self.assertEqual(selected.population_node_id, root.population_node_id)
                else:
                    self.assertIsNone(selected)
                    with self.assertRaises(RuntimeError):
                        method.expand_node(root, population, experiences)
                method._checkpoint_index += 1
                method._write_checkpoint()
                checkpoint = Path(directory) / f"checkpoint_{method._checkpoint_index:06d}.json"
                with self.assertRaisesRegex(ValueError, "allow_recipe_repeats"):
                    build(not repeat).restore(str(checkpoint))

    def test_node_persistence_parent_context_reward_and_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            encoder = Encoder()
            config = {"enabled": True, "dimensions": 2, "model": "fake"}

            def expand(population, recipe, selection_num, target_size,
                       on_evaluated, reserve_sample_order):
                child = individual(3)
                child._recipe_sample_order = reserve_sample_order()
                on_evaluated(child)
                return [child]

            def build(embedding_config=config):
                return MCTSRecipe(
                    recipes={"a": {"comparison": True}, "b": {"comparison": True}},
                    pop_size=2, selection_num=1, store_dir=directory,
                    expand_fn=expand, max_sample_count=100,
                    adaptive_config={"enabled": True, "recipes_per_expansion": 1},
                    node_embedding_model=encoder,
                    node_embedding_config=embedding_config)

            method = build()
            initial = [individual(1), individual(2)]
            for function in initial:
                function._recipe_sample_order = method.reserve_sample_order()
            root = method.initialize(initial)
            population, experiences = method._load_node_state(root.population_node_id)
            parent_context = method._adaptive_state(root, population)
            np.testing.assert_allclose(parent_context, [1.5, 3])
            children = method.expand_node(root, population, experiences)
            self.assertEqual(len(children), 1)
            child = children[0]
            # best gain=1, mean gain=1 => reward=1; update uses parent context.
            np.testing.assert_allclose(
                method.adaptive_router.b[child.incoming_recipe_id], [1, 1.5, 3])
            features = np.asarray([[1, 1, 2], [1, 2, 4]])
            np.testing.assert_allclose(
                method.adaptive_router.A[child.incoming_recipe_id],
                np.eye(3) + features.T @ features / 2)
            self.assertEqual(method.adaptive_router.trials[child.incoming_recipe_id], 1)
            method._write_disk_tree_transition(root, children)
            method.store.flush()
            method._commit_pending_tree_state()
            method._checkpoint_index += 1
            method._write_checkpoint()
            record = method.store.load_node(child.population_node_id)
            np.testing.assert_allclose(record["embedding"], [2.5, 5])
            self.assertEqual(len(record["recipe_selection"]["individual_votes"]), 2)
            self.assertEqual(record["recipe_selection"]["selected_ids"], [child.incoming_recipe_id])
            self.assertTrue(all("generation_raw_response" not in item
                                and "embedding" in item for item in record["algorithms"]))
            call_count = len(encoder.calls)
            restored = build()
            checkpoint = Path(directory) / f"checkpoint_{method._checkpoint_index:06d}.json"
            restored.restore(str(checkpoint))
            population, _ = restored._load_node_state(child.population_node_id)
            np.testing.assert_allclose(restored._adaptive_state(child, population), [2.5, 5])
            self.assertEqual(len(encoder.calls), call_count)
            np.testing.assert_allclose(restored._individual_contexts(population), [[3, 6], [2, 4]])
            with self.assertRaises(ValueError):
                build({**config, "dimensions": 3}).restore(str(checkpoint))


if __name__ == "__main__":
    unittest.main()
