
params = [[0,1,2]]

def helper(func):
	def wrapper():
		n1,n2,n3 = params[i]
		return func(n1,n2,n3)

