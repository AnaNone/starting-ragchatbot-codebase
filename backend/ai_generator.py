import anthropic
from typing import List, Optional

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # System prompt template - filled in with the configured tool-round budget
    SYSTEM_PROMPT_TEMPLATE = """ You are an AI assistant specialized in course materials and educational content with access to a comprehensive search tool for course information.

Tool Usage:
- **get_course_outline**: Use for outline, overview, or "what does course X cover?" queries
- **search_course_content**: Use for specific topic or content questions within a course
- **Sequential tool use is allowed**: you may call tools across up to {max_tool_rounds} steps when needed
  (e.g., look up a course outline, then search within a specific lesson it reveals; or refine a
  search with different terms if the first attempt is unhelpful)
- **Be efficient**: once you have enough information to answer, stop calling tools and respond
- Synthesize tool results into accurate, fact-based responses
- If a tool yields no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without searching
- **Course-specific questions**: Search first, then answer
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, search explanations, or question-type analysis
 - Do not mention "based on the search results"


All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""

    def __init__(self, api_key: str, model: str, max_tool_rounds: int = 2):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.max_tool_rounds = max_tool_rounds
        self.system_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(max_tool_rounds=max_tool_rounds)

        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None) -> str:
        """
        Generate AI response, allowing Claude to call tools across multiple
        sequential rounds (up to self.max_tool_rounds) before producing a
        final answer.

        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools

        Returns:
            Generated response as string
        """

        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.system_prompt}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history
            else self.system_prompt
        )

        messages = [{"role": "user", "content": query}]
        round_num = 0

        while True:
            api_params = {
                **self.base_params,
                "messages": messages,
                "system": system_content
            }

            # Only offer tools while we still have rounds left to use them.
            # Omitting tools on the final round forces Claude to produce a
            # text answer (stop_reason can no longer be "tool_use").
            if tools and round_num < self.max_tool_rounds:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}

            response = self.client.messages.create(**api_params)

            # Claude is done (no further tool use requested) - return its answer
            if response.stop_reason != "tool_use" or not tool_manager:
                return response.content[0].text

            round_num += 1
            messages.append({"role": "assistant", "content": response.content})

            tool_results, fatal_error = self._execute_tool_round(response, tool_manager)
            if not tool_results:
                return "I was unable to process your request."
            messages.append({"role": "user", "content": tool_results})

            # On a hard tool failure, stop looping and let Claude give a
            # best-effort answer immediately using whatever context it has.
            if fatal_error:
                final_params = {
                    **self.base_params,
                    "messages": messages,
                    "system": system_content
                }
                final_response = self.client.messages.create(**final_params)
                return final_response.content[0].text

    def _execute_tool_round(self, response, tool_manager):
        """
        Execute every tool_use block in a response, returning the tool_result
        content blocks to feed back to Claude plus a flag indicating whether
        an unexpected (hard) error occurred during execution.

        Soft errors (e.g. "Tool 'X' not found", empty search results) are
        returned by tool_manager.execute_tool() as normal string content and
        flow back to Claude like any other result, so it can adjust course.
        Hard errors (exceptions raised by the tool) are caught here, surfaced
        to Claude via the is_error flag, and signal the caller to stop
        looping and wrap up with a best-effort answer.

        Returns:
            (tool_results, fatal_error) tuple
        """
        tool_results = []
        fatal_error = False

        for content_block in response.content:
            if content_block.type != "tool_use":
                continue

            try:
                tool_result = tool_manager.execute_tool(
                    content_block.name,
                    **content_block.input
                )
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
            except Exception as e:
                fatal_error = True
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": f"Tool execution failed: {e}",
                    "is_error": True
                })

        return tool_results, fatal_error