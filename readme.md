# Graduate AI Final Project/Exam 

Responses: https://github.com/tejpshah/cs520-final-tejshah/tree/main/writeups

Question 1.1, 1.2, & 1.3 deals with localization. Suppose that you have a robot in a nuclear reactor that has blocked walls and unblocked walls. While you do not know where the robot is, you can issue a sequence of commands and know that the robot will execute that command. If the command makes the robot go out of bounds or into a wall, then the robot will stay in place. The goal is to perfectly localize the robot efficiently after issuing a sequence of commands. Our goal is to localize the robot after a sequence of actions. Initially, our probability distribution over the unblocked cells is uniform. Formally, we want to reach a terminal state where the probability distribution has a 1.0 at one of the unblocked cells and a probability of 0.0 everywhere else. Formally, we want to move the belief state about the robot’s location from the region of high entropy to the region of lowest entropy. I formulate this as a MDP and hand-craft a policy with utilities/rewards based on multiple objectives, using a two-step forward lookahead. For Q1.4, I outline a method to find the worst nuclear reactor configuration using genetic algorithms. Question 2.1 deals with dealing with what's possible and what's not possible in a short clip of the movie of War Games. Question 2.2.1 outlines why LLMs like Galactica or GPT hallucinate incorrect responses to prompts. Question 2.2.2 outlines a system using RLHF to develop a model for better prompt engineering LLMs like ChatGPT.

# The Best Canvas Announcement You Will Ever Read

## "To Clarify My Previous Announcement"
"both of those were outputs from ChatGPT.

I think they're interesting to consider, for two reasons:

First - are you really making the best use of your time here at Rutgers, here on this earth, letting the world's most advanced autocomplete do your work for you? Are you growing as a person, are you learning, gaining anything of value? Are you gaining anything of value that you'll take to your next job, into your future? And if your immediate reaction is "yes, an A" - I want you to think back on this moment many years from now when you are sitting at your desk looking at the resume and transcript of a job applicant, looking at their long record of As, and I want you to ask yourself if any of that really means anything.

Second - do you really think that either of those essays were that good? After reading them, do you think you really had a good notion of how to sit down and actually build the system that I was looking for? Or was it more just hitting the vocabulary, the terms that it 'knew' were supposed to go in an answer like this? You know an ML problem is going to need data. It doesn't really help you at all to be told that. You know that your model is going to have to map or operate on your data somehow. It doesn't help you to be told that. What does your data /look/ like? How big is it? Where does it come from? How can you use it? What kind of operations are /meaningful/ on this data? What does loss /mean/ on a task like this? Surely we can use gradient descent to train our model, but this is like saying we can earn more money by minimizing our losses. True, but useless. Form without function.

Perhaps the musings of one professor don't amount to a hill of beans in this crazy world, but this is my hill, and these are my beans. As a mathematician, I tend to abstract and generalize, perhaps too far, and certainly further than I know many of you were interested in. But both of the above points lead me to the following - all I ask, as we move into the new year, is for you to take a moment and ask, why are you here, and who do you want to be?"

# Academic Integrity
Please follow both Rutgers University's Principles of Academic Integrity and the Rutgers Department of Computer Science's Academic Integrity Policy.

