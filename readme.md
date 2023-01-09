# Graduate AI Final Project/Exam 

Responses: https://github.com/tejpshah/cs520-final-tejshah/tree/main/writeups

Question 1.1, 1.2, & 1.3 deals with localization. Suppose that you have a robot in a nuclear reactor that has blocked walls and unblocked walls. While you do not know where the robot is, you can issue a sequence of commands and know that the robot will execute that command. If the command makes the robot go out of bounds or into a wall, then the robot will stay in place. The goal is to perfectly localize the robot efficiently after issuing a sequence of commands. Our goal is to localize the robot after a sequence of actions. Initially, our probability distribution over the unblocked cells is uniform. Formally, we want to reach a terminal state where the probability distribution has a 1.0 at one of the unblocked cells and a probability of 0.0 everywhere else. Formally, we want to move the belief state about the robot’s location from the region of high entropy to the region of lowest entropy. I formulate this as a MDP and hand-craft a policy with utilities/rewards based on multiple objectives, using a two-step forward lookahead. For Q1.4, I outline a method to find the worst nuclear reactor configuration using genetic algorithms. Question 2.1 deals with dealing with what's possible and what's not possible in a short clip of the movie of War Games. Question 2.2.1 outlines why LLMs like Galactica or GPT hallucinate incorrect responses to prompts. Question 2.2.2 outlines a system using RLHF to develop a model for better prompt engineering LLMs like ChatGPT.

# Academic Integrity
Please follow both Rutgers University's Principles of Academic Integrity and the Rutgers Department of Computer Science's Academic Integrity Policy.

