#!/usr/bin/env python3
"""
End-to-End MCP Server Test

Tests the complete MCP protocol communication by starting the server
as a subprocess and connecting via the MCP SDK client.

Note: Run `uv run python ingest.py` first to populate the knowledge base.
"""

import json
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_search_knowledge():
    """Test semantic search via MCP protocol."""
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "python", "rag_knowledge_mcp.py"],
        env=None,
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # Call the search tool
            result = await session.call_tool(
                "rag_search_knowledge",
                arguments={
                    "params": {
                        "query": "machine learning",
                        "top_k": 3,
                        "score_threshold": 0.0,
                        "include_metadata": True,
                        "response_format": "json",
                    }
                },
            )

            # Verify response
            assert result.content, "Search should return content"
            assert len(result.content) > 0

            if result.isError:
                pytest.fail(f"Tool returned error: {result.content[0].text}")

            search_text = result.content[0].text
            search_results = json.loads(search_text)

            assert "query" in search_results
            assert "results_count" in search_results
            assert "results" in search_results
            assert search_results["query"] == "machine learning"
