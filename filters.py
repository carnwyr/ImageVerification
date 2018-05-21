'''
Module that contains filters used in SRM
They are used to initialize the first layer of the CNN
'''

import numpy as np

# return np array accepted as wight matrix by convolutional layer
def get_filters_as_weights():
	first_1 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[1]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[-1]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	first_2 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[1]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[-1]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	first_3 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[1]], [[0]] ],
						[ [[0]], [[0]], [[-1]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	first_4 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[-1]], [[1]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	first_5 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[-1]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[1]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	first_6 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[-1]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[1]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	first_7 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[-1]], [[0]], [[0]] ],
						[ [[0]], [[1]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	first_8 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[1]], [[-1]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])


	second_1 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[1]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[-2]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[1]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	second_2 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[1]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[-2]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[1]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	second_3 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[1]], [[0]] ],
						[ [[0]], [[0]], [[-2]], [[0]], [[0]] ],
						[ [[0]], [[1]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	second_4 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[1]], [[-2]], [[1]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])


	third_1 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[1]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[-3]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[3]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[-1]] ] ])

	third_2 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[1]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[-3]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[3]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[-1]], [[0]], [[0]] ] ])

	third_3 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[1]], [[0]] ],
						[ [[0]], [[0]], [[-3]], [[0]], [[0]] ],
						[ [[0]], [[3]], [[0]], [[0]], [[0]] ],
						[ [[-1]], [[0]], [[0]], [[0]], [[0]] ] ])

	third_4 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[-1]], [[3]], [[-3]], [[1]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	third_5 = np.array([[ [[-1]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[3]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[-3]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[1]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	third_6 = np.array([[ [[0]], [[0]], [[-1]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[3]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[-3]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[1]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	third_7 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[-1]] ],
						[ [[0]], [[0]], [[0]], [[3]], [[0]] ],
						[ [[0]], [[0]], [[-3]], [[0]], [[0]] ],
						[ [[0]], [[1]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	third_8 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[1]], [[-3]], [[3]], [[-1]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						[ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])


	square_3 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						 [ [[0]], [[-1]], [[2]], [[-1]], [[0]] ],
						 [ [[0]], [[2]], [[-4]], [[2]], [[0]] ],
						 [ [[0]], [[-1]], [[2]], [[-1]], [[0]] ],
						 [ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])


	edge_3_1 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						 [ [[0]], [[-1]], [[2]], [[-1]], [[0]] ],
						 [ [[0]], [[2]], [[-4]], [[2]], [[0]] ],
						 [ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						 [ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	edge_3_2 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						 [ [[0]], [[0]], [[2]], [[-1]], [[0]] ],
						 [ [[0]], [[0]], [[-4]], [[2]], [[0]] ],
						 [ [[0]], [[0]], [[2]], [[-1]], [[0]] ],
						 [ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	edge_3_3 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						 [ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						 [ [[0]], [[2]], [[-4]], [[2]], [[0]] ],
						 [ [[0]], [[-1]], [[2]], [[-1]], [[0]] ],
						 [ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	edge_3_4 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						 [ [[0]], [[-1]], [[2]], [[0]], [[0]] ],
						 [ [[0]], [[2]], [[-4]], [[0]], [[0]] ],
						 [ [[0]], [[-1]], [[2]], [[0]], [[0]] ],
						 [ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])


	square_5 = np.array([[ [[-1]], [[2]], [[-2]], [[2]], [[-1]] ],
						 [ [[2]], [[-6]], [[8]], [[-6]], [[2]] ],
						 [ [[-2]], [[8]], [[-12]], [[8]], [[-2]] ],
						 [ [[2]], [[-6]], [[8]], [[-6]], [[2]] ],
						 [ [[-1]], [[2]], [[-2]], [[2]], [[-1]] ] ])


	edge_5_1 = np.array([[ [[-1]], [[2]], [[-2]], [[2]], [[-1]] ],
						 [ [[2]], [[-6]], [[8]], [[-6]], [[2]] ],
						 [ [[-2]], [[8]], [[-12]], [[8]], [[-2]] ],
						 [ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						 [ [[0]], [[0]], [[0]], [[0]], [[0]] ] ])

	edge_5_2 = np.array([[ [[0]], [[0]], [[-2]], [[2]], [[-1]] ],
						 [ [[0]], [[0]], [[8]], [[-6]], [[2]] ],
						 [ [[0]], [[0]], [[-12]], [[8]], [[-2]] ],
						 [ [[0]], [[0]], [[8]], [[-6]], [[2]] ],
						 [ [[0]], [[0]], [[-2]], [[2]], [[-1]] ] ])

	edge_5_3 = np.array([[ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						 [ [[0]], [[0]], [[0]], [[0]], [[0]] ],
						 [ [[-2]], [[8]], [[-12]], [[8]], [[-2]] ],
						 [ [[2]], [[-6]], [[8]], [[-6]], [[2]] ],
						 [ [[-1]], [[2]], [[-2]], [[2]], [[-1]] ] ])

	edge_5_4 = np.array([[ [[-1]], [[2]], [[-2]], [[0]], [[0]] ],
						 [ [[2]], [[-6]], [[8]], [[0]], [[0]] ],
						 [ [[-2]], [[8]], [[-12]], [[0]], [[0]] ],
						 [ [[2]], [[-6]], [[8]], [[0]], [[0]] ],
						 [ [[-1]], [[2]], [[-2]], [[0]], [[0]] ] ])

	kernels = [first_1,first_2,first_3,first_4,first_5,first_6,first_7,first_8,second_1,second_2,second_3,second_4,third_1,third_2,third_3,third_4,third_5,third_6,third_7,third_8,square_3,edge_3_1,edge_3_2,edge_3_3,edge_3_4,square_5,edge_5_1,edge_5_2,edge_5_3,edge_5_4]
	stacks = []

	for j in range(1,31):
		k = (j - 1) % 10 + 1
		stack = np.concatenate((kernels[3*k-3], kernels[3*k-2]), axis=2)
		stack = np.concatenate((stack, kernels[3*k-1]), axis=2)
		stacks.append(stack)

	filters = np.concatenate(stacks, axis=3)

	return filters