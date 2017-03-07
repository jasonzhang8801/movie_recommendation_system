from __future__ import division
user_review_matrix = []
def parse_eval_test():
	i = 151
	with open("predict.txt") as lines:
		for line in lines:
			review = map(int, line.split("\t"))
			user_review_matrix.append(review)
			r_count = 5
			p_count = 10
			for j in range(len(review)):
				if review[j] != 0 and r_count > 0:
					# print i, " ", j+1, " ", review[j]
					r_count -= 1
				if(r_count < 1):
					j += 1
					break	
			for k in range(j, len(review)):
				if review[k] != 0 and p_count > 0:
					# print i, " ", k+1, review[k]
					p_count -= 1
			i += 1

def get_MAE():
	predict = []
	actual = []
	with open("eval_result.txt") as lines:
		for line in lines:
			data = line.split(" ")
			predict.append(int(data[2]))

	with open("ground2.txt") as lines:
		for line in lines:
			data = line.split(" ")
			actual.append(int(data[2]))

	total = 0
	for i in range(len(predict)):
		total += abs(predict[i] - actual[i])

	print "MAE is ", float(total/500)


parse_eval_test()
get_MAE()