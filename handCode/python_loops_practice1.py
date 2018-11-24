#Python loops: FOR
fruits = ["apple", "banana","cherry"]
for x in fruits:
	print(x)

#Python loops: Loop through strings
for x in "banana":
	print(x)

#python loops: Using the break statement
for x in fruits:
	print(x)
	if x == "banana":
		break


#python loops: Using the break statement and print
for x in fruits:
	if x == "banana":
		break
	
	print(x)

#python loops: Using the continue statement
for x in fruits:
	if x == "banana":
		continue
	print(x)

#python loops: using range
for x in range(6):
	print(x)


#python loops: using range with paramaters
for x in range(2,6):
	print(x)

#python loops: using range with paramenters and default
for x in range(2,30,3):
	print(x)

#python loops: using the Else statement
for x in range(6):
	print(x)
else:
	print("Finally finished!")


#python loops: nested loops
adj = ["red","big","tasty"]
fruits = ["apple","banana","cherry"]

for x in adj:
	for y in fruits:
		print(x,y)


#python loops: recursion
def tri_recursion(k):
	if(k>0):
		result = k+tri_recursion(K-1)
		print(result)
	else:
		result = 0
	return result

print("\n\nRecursion Example Results")
tri_recursion(6)