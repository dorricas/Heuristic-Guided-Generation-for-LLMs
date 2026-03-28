sort_example = """Consider the task of sorting to generate a list in ascending order. Use comparsion to determine the ordering of numbers. At each step you are allowed to make one swap to produce a sorted list.
 
Input: -1, -1, 2, -2, 0  
Steps:  
Swap 2 and -2 : [-1, -1, -2, 2, 0]  
Swap 2 and 0 : [-1, -1, -2, 0, 2]  
Swap -2 and -1 : [-1, -2, -1, 0, 2]  
Swap -2 and -1 : [-2, -1, -1, 0, 2]  
Answer: [-2, -1, -1, 0, 2]  

Input: 9, 3.5, 7, 3.4, 0  
Steps:  
Swap 9 and 3.5 : [3.5, 9, 7, 3.4, 0]  
Swap 9 and 7 : [3.5, 7, 9, 3.4, 0]  
Swap 9 and 3.4 : [3.5, 7, 3.4, 9, 0]  
Swap 9 and 0 : [3.5, 7, 3.4, 0, 9]  
Swap 3.4 and 7 : [3.5, 3.4, 7, 0, 9]  
Swap 3.4 and 3.5 : [3.4, 3.5, 7, 0, 9]  
Swap 0 and 7 : [3.4, 3.5, 0, 7, 9]  
Swap 0 and 3.5 : [3.4, 0, 3.5, 7, 9]  
Swap 0 and 3.4 : [0, 3.4, 3.5, 7, 9]  
Answer: [0, 3.4, 3.5, 7, 9]  

Input: -4.5, -3.3, -2.1, -1.1, 0  
Steps: 
Answer: [-4.5, -3.3, -2.1, -1.1, 0]  
  
Input: 1, 3, 2, 5, 4  
Steps:  
Swap 3 and 2 : [1, 2, 3, 5, 4]  
Swap 5 and 4 : [1, 2, 3, 4, 5]  
Answer: [1, 2, 3, 4, 5]
"""
