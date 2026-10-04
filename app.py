import json
import logging
import asyncio
from typing import Dict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

# Import state and stages
from stages import (
    WorkflowState,
    stage_1_intake,
    stage_2_understand,
    stage_3_prepare,
    stage_4_ask,
    stage_5_wait,
    stage_6_retrieve,
    stage_7_decide,
    stage_8_update,
    stage_9_create,
    stage_10_do,
    stage_11_complete,
)

# Re-export MCP classes for convenience and backwards compatibility
from mcp_client import MCPServer, MCPClient, common_client, atlas_client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============= LangGraph Workflow Builder =============
def build_customer_support_workflow():
    """Build the LangGraph workflow with 11 stages"""
    
    # Create workflow graph
    workflow = StateGraph(WorkflowState)
    
    # Add all nodes (stages)
    workflow.add_node("intake", stage_1_intake)
    workflow.add_node("understand", stage_2_understand)
    workflow.add_node("prepare", stage_3_prepare)
    workflow.add_node("ask", stage_4_ask)
    workflow.add_node("wait", stage_5_wait)
    workflow.add_node("retrieve", stage_6_retrieve)
    workflow.add_node("decide", stage_7_decide)
    workflow.add_node("update", stage_8_update)
    workflow.add_node("create", stage_9_create)
    workflow.add_node("do", stage_10_do)
    workflow.add_node("complete", stage_11_complete)
    
    # Define edges (flow between stages)
    workflow.add_edge("intake", "understand")
    workflow.add_edge("understand", "prepare")
    workflow.add_edge("prepare", "ask")
    workflow.add_edge("ask", "wait")
    workflow.add_edge("wait", "retrieve")
    workflow.add_edge("retrieve", "decide")
    workflow.add_edge("decide", "update")
    workflow.add_edge("update", "create")
    workflow.add_edge("create", "do")
    workflow.add_edge("do", "complete")
    workflow.add_edge("complete", END)
    
    workflow.set_entry_point("intake")
    
    return workflow.compile()


# ============= Main Execution =============
async def run_customer_support_agent(input_payload: Dict):
    print("\n" + "="*60)
    print("LANGIE - Customer Support Agent Starting")
    print("="*60 + "\n")
    
    # Initialize state
    initial_state = WorkflowState(
        customer_name=input_payload.get('customer_name', ''),
        email=input_payload.get('email', ''),
        query=input_payload.get('query', ''),
        priority=input_payload.get('priority', ''),
        ticket_id=input_payload.get('ticket_id', ''),
        parsed_intent=None,
        entities=None,
        sentiment=None,
        normalized_fields=None,
        enriched_data=None,
        flags_calculations=None,
        clarification_needed=None,
        clarification_question=None,
        customer_response=None,
        knowledge_base_results=None,
        relevant_data=None,
        solutions=None,
        chosen_solution=None,
        escalation_required=None,
        ticket_status=None,
        generated_response=None,
        executed_actions=None,
        notifications_sent=None,
        final_status=None,
        processing_time=None,
        stage_history=[],
        current_stage='',
        timestamp=''
    )
    
    # Build workflow
    workflow = build_customer_support_workflow()
    
    # Execute workflow
    final_state = await workflow.ainvoke(initial_state)
    
    print("\n" + "="*60)
    print("✅ WORKFLOW EXECUTION COMPLETE")
    print("="*60 + "\n")
    
    return final_state


async def demo():
    sample_input = {
        "customer_name": "John Smith",
        "email": "",
        "query": "When will my order come? My order id is #456 and my last name is Smith",
        "priority": "medium",
        "ticket_id": "T001"
    }
    
    print("📝 Input Payload:")
    print(json.dumps(sample_input, indent=2))
    print("\n" + "="*60 + "\n")
    
    result = await run_customer_support_agent(sample_input)
    print()


if __name__ == "__main__":
    asyncio.run(demo())