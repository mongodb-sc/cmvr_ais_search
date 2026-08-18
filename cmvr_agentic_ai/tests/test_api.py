from __future__ import annotations

import json
import unittest
from unittest.mock import MagicMock, patch

from agent.loop import AgentRun, ToolCall
from api import _event_stream, _save_history


class ApiStreamTests(unittest.TestCase):
    def test_stream_emits_tool_event_before_final_answer(self) -> None:
        tool_call = ToolCall(
            name="cmvr_search",
            input={"query": "bus tests"},
            result={"rules": [{"ruleNumber": "124"}], "ais_codes": ["AIS-052"]},
        )

        def fake_run_agent(query: str, *, on_tool_call, verbose: bool) -> AgentRun:
            self.assertEqual(query, "bus tests")
            self.assertFalse(verbose)
            on_tool_call(tool_call)
            return AgentRun(
                answer="Rule 124 applies.",
                tool_calls=[tool_call],
                turns=2,
                stopped_reason="end_turn",
            )

        with (
            patch("api.run_agent", side_effect=fake_run_agent),
            patch("api._save_history") as save_history,
        ):
            events = [json.loads(line) for line in _event_stream("bus tests")]

        self.assertEqual([event["type"] for event in events], ["tool", "done"])
        self.assertEqual(events[0]["name"], "cmvr_search")
        self.assertEqual(events[0]["result"]["ais_codes"], ["AIS-052"])
        self.assertEqual(events[1]["answer"], "Rule 124 applies.")
        self.assertEqual(events[1]["toolCalls"], 1)
        save_history.assert_called_once()
        self.assertEqual(save_history.call_args.args[0], "bus tests")
        self.assertEqual(save_history.call_args.args[2][0]["name"], "cmvr_search")

    def test_save_history_deletes_entries_beyond_newest_five(self) -> None:
        collection = MagicMock()
        collection.find.return_value.sort.return_value.skip.return_value = [
            {"_id": "old-1"},
            {"_id": "old-2"},
        ]
        result = AgentRun(
            answer="Rule 124 applies.",
            tool_calls=[],
            turns=2,
            stopped_reason="end_turn",
        )

        with patch("api.config.history_collection", return_value=collection):
            _save_history("bus tests", result, [])

        inserted = collection.insert_one.call_args.args[0]
        self.assertEqual(inserted["query"], "bus tests")
        self.assertEqual(inserted["answer"], "Rule 124 applies.")
        collection.find.return_value.sort.assert_called_once_with("createdAt", -1)
        collection.find.return_value.sort.return_value.skip.assert_called_once_with(5)
        collection.delete_many.assert_called_once_with(
            {"_id": {"$in": ["old-1", "old-2"]}}
        )


if __name__ == "__main__":
    unittest.main()
