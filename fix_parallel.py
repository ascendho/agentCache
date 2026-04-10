import re
with open('agent/nodes.py', 'r') as f:
    content = f.read()

import_pattern = r'from typing import List, Dict, Any, Optional, TypedDict'
if 'concurrent.futures' not in content:
    content = content.replace(import_pattern, import_pattern + '\nimport concurrent.futures')

research_pattern = """
    try:
        for sub_question, is_cached in cache_hits.items():
            if not is_cached:  # Research cache misses or inadequate previous research
                current_iteration = research_iterations.get(sub_question, 0)
                strategy = current_strategies.get(sub_question, "initial")

                logger.info(
                    f"🔍 Researching: '{sub_question[:50]}...' (iteration {current_iteration + 1}, strategy: {strategy})"
                )

                # Adapt research strategy based on iteration and feedback
                research_prompt = sub_question
                if current_iteration > 0 and feedback.get(sub_question):
                    research_prompt = f\"\"\"
                    Previous research was insufficient. Feedback: {feedback[sub_question]}
                    
                    Original question: {sub_question}
                    
                    Please research this more thoroughly, focusing on the specific improvements mentioned in the feedback.
                    Use different search terms and approaches than before.
                    \"\"\"

                # Use ReAct agent with enhanced prompting
                research_result = researcher_agent.invoke(
                    {"messages": [HumanMessage(content=research_prompt)]}
                )

                # Track research LLM usage (ReAct agent makes multiple internal calls)
                llm_calls["research_llm"] = llm_calls.get("research_llm", 0) + 1

                # Extract the final answer from the agent's response
                if research_result and "messages" in research_result:
                    answer = research_result["messages"][-1].content
                    sub_answers[sub_question] = answer
                    questions_researched += 1

                    # Update iteration count and strategy
                    research_iterations[sub_question] = current_iteration + 1
                    current_strategies[sub_question] = (
                        f"iteration_{current_iteration + 1}"
                    )

                    logger.info(
                        f"   ✅ Research complete (iteration {current_iteration + 1}): '{answer[:60]}...'"
                    )
                    # Note: Caching happens after quality validation, not here
                else:
                    logger.warning(
                        f"   ❌ No response from researcher for: {sub_question}"
                    )
                    sub_answers[sub_question] = (
                        "I couldn't find specific information about this. Please contact support for detailed assistance."
                    )
"""

research_replacement = """
    try:
        def process_research(sub_question):
            current_iteration = research_iterations.get(sub_question, 0)
            strategy = current_strategies.get(sub_question, "initial")
            logger.info(f"🔍 Researching: '{sub_question[:50]}...' (iteration {current_iteration + 1}, strategy: {strategy})")
            
            research_prompt = sub_question
            if current_iteration > 0 and feedback.get(sub_question):
                research_prompt = f\"\"\"
                Previous research was insufficient. Feedback: {feedback[sub_question]}
                Original question: {sub_question}
                Please research this more thoroughly, focusing on the specific improvements mentioned in the feedback.
                Use different search terms and approaches than before.
                \"\"\"
            
            research_result = researcher_agent.invoke({"messages": [HumanMessage(content=research_prompt)]})
            
            if research_result and "messages" in research_result:
                answer = research_result["messages"][-1].content
                return sub_question, "success", answer, current_iteration
            else:
                return sub_question, "failure", "I couldn't find specific information about this.", current_iteration

        tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for sub_question, is_cached in cache_hits.items():
                if not is_cached:
                    tasks.append(executor.submit(process_research, sub_question))
            
            for future in concurrent.futures.as_completed(tasks):
                sub_question, status, answer, current_iteration = future.result()
                llm_calls["research_llm"] = llm_calls.get("research_llm", 0) + 1
                if status == "success":
                    sub_answers[sub_question] = answer
                    questions_researched += 1
                    research_iterations[sub_question] = current_iteration + 1
                    current_strategies[sub_question] = f"iteration_{current_iteration + 1}"
                    logger.info(f"   ✅ Research complete (iteration {current_iteration + 1}): '{answer[:60]}...'")
                else:
                    logger.warning(f"   ❌ No response from researcher for: {sub_question}")
                    sub_answers[sub_question] = answer
"""

if research_pattern.strip() in content:
    content = content.replace(research_pattern, research_replacement)
else:
    print("Research pattern not found")

eval_pattern = """
    try:
        for sub_question in sub_questions:
            # Skip cached answers - they're already validated
            if cache_hits.get(sub_question, False):
                quality_scores[sub_question] = 1.0  # Cached answers are high quality
                continue

            # Skip if no answer yet
            if sub_question not in sub_answers:
                needs_more_research.append(sub_question)
                continue

            answer = sub_answers[sub_question]
            current_iteration = state.get("research_iterations", {}).get(
                sub_question, 0
            )

            # Evaluate research quality
            evaluation_prompt = f\"\"\"
            Evaluate the quality and completeness of this research result for answering the user's question.
            
            Original sub-question: {sub_question}
            Research result: {answer}
            Research iteration: {current_iteration + 1}
            
            Provide:
            1. A quality score from 0.0 to 1.0 (where 1.0 is perfect, 0.7+ is adequate)
            2. Brief feedback on what's missing or could be improved (if score < 0.7)
            
            Format your response as:
            SCORE: 0.X
            FEEDBACK: [your feedback or "Adequate" if score >= 0.7]
            \"\"\"

            evaluation = research_llm.invoke([HumanMessage(content=evaluation_prompt)])

            # Track LLM usage
            llm_calls["research_llm"] = llm_calls.get("research_llm", 0) + 1

            # Parse the evaluation
            try:
                lines = evaluation.content.strip().split("\\n")
                score_line = [l for l in lines if l.startswith("SCORE:")][0]
                feedback_line = [l for l in lines if l.startswith("FEEDBACK:")][0]

                score = float(score_line.split("SCORE:")[1].strip())
                feedback_text = feedback_line.split("FEEDBACK:")[1].strip()

                quality_scores[sub_question] = score
                feedback[sub_question] = feedback_text

                if score < 0.7:
                    logger.info(
                        f"   🔄 {sub_question[:40]}... - Score: {score:.2f} (Needs improvement)"
                    )
                else:
                    logger.info(
                        f"   ✅ {sub_question[:40]}... - Score: {score:.2f} - Adequate"
                    )

            except Exception as parse_error:
                logger.warning(
                    f"Failed to parse evaluation for {sub_question}: {parse_error}"
                )
                # Default to adequate if we can't parse
                quality_scores[sub_question] = 0.8
                feedback[sub_question] = "Evaluation parsing failed - assuming adequate"
"""

eval_replacement = """
    try:
        def evaluate_sub_question(sub_question, answer, current_iteration):
            evaluation_prompt = f\"\"\"
            Evaluate the quality and completeness of this research result for answering the user's question.
            
            Original sub-question: {sub_question}
            Research result: {answer}
            Research iteration: {current_iteration + 1}
            
            Provide:
            1. A quality score from 0.0 to 1.0 (where 1.0 is perfect, 0.7+ is adequate)
            2. Brief feedback on what's missing or could be improved (if score < 0.7)
            
            Format your response as:
            SCORE: 0.X
            FEEDBACK: [your feedback or "Adequate" if score >= 0.7]
            \"\"\"
            
            try:
                evaluation = research_llm.invoke([HumanMessage(content=evaluation_prompt)])
                lines = evaluation.content.strip().split("\\n")
                score_line = [l for l in lines if l.startswith("SCORE:")][0]
                feedback_line = [l for l in lines if l.startswith("FEEDBACK:")][0]
                
                score = float(score_line.split("SCORE:")[1].strip())
                feedback_text = feedback_line.split("FEEDBACK:")[1].strip()
                return sub_question, "success", score, feedback_text
            except Exception as e:
                return sub_question, "error", 0.8, f"Evaluation parsing failed: {e}"

        eval_tasks = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            for sub_question in sub_questions:
                if cache_hits.get(sub_question, False):
                    quality_scores[sub_question] = 1.0
                    continue
                if sub_question not in sub_answers:
                    needs_more_research.append(sub_question)
                    continue
                
                answer = sub_answers[sub_question]
                current_iteration = state.get("research_iterations", {}).get(sub_question, 0)
                eval_tasks.append(executor.submit(evaluate_sub_question, sub_question, answer, current_iteration))

            for future in concurrent.futures.as_completed(eval_tasks):
                sub_question, status, score, feedback_text = future.result()
                llm_calls["research_llm"] = llm_calls.get("research_llm", 0) + 1
                
                quality_scores[sub_question] = score
                feedback[sub_question] = feedback_text
                
                if status == "success":
                    if score < 0.7:
                        logger.info(f"   🔄 {sub_question[:40]}... - Score: {score:.2f} (Needs improvement)")
                    else:
                        logger.info(f"   ✅ {sub_question[:40]}... - Score: {score:.2f} - Adequate")
                else:
                    logger.warning(f"Failed to parse evaluation for {sub_question} - assuming adequate")
"""

if eval_pattern.strip() in content:
    content = content.replace(eval_pattern, eval_replacement)
else:
    print("Eval pattern not found")

with open('agent/nodes.py', 'w') as f:
    f.write(content)
