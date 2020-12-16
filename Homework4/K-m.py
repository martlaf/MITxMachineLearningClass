import numpy as np

x = np.array([[0, -6], [4, 4], [0, 0], [-5, 2]])
z = np.array([[-5, 2], [0, -6]])
assignment = np.zeros(x.shape[0])


def assign_points(x, z):
    distances = np.zeros([x.shape[0], z.shape[0]])
    for i in np.arange(x.shape[0]):
        for k in np.arange(z.shape[0]):
            distances[i, k] = np.linalg.norm(x[i,:] - z[k,:], 1)
    return np.argmin(distances, axis=1)


def find_medoids(x, z, assignment):
    K = z.shape[0]
    for k in np.arange(K):
        nk = np.count_nonzero(assignment==k)
        selected_points = x[assignment==k]
        tries = np.zeros([nk, x.shape[1]])
        for i in np.arange(nk):
            tries[i,:] = np.sum(selected_points - selected_points[i,:], axis=0)

        z[k,:] = x[assignment==k][np.argmin(np.linalg.norm(tries, 2, axis=1)),:]

        #mean = np.sum(x[assignment==k], axis=0)/nk
        #z[k,:] = x[np.argmin(np.linalg.norm(x-mean, 1, axis=1))]
    return z




assignment = assign_points(x, z)
#print(assignment)
z = find_medoids(x, z, assignment)
#print(z)
assignment = assign_points(x, z)
#print(assignment)
z = find_medoids(x, z, assignment)
#print(z)
assignment = assign_points(x, z)
#print(assignment)
z = find_medoids(x, z, assignment)
#print(z)

def find_centroids(x, z, assignment):
    K = z.shape[0]
    for k in np.arange(K):
        nk = np.count_nonzero(assignment==k)
        z[k,:] = np.sum(x[assignment==k], axis=0)/nk
    return z


x = np.array([[0., -6.], [4., 4.], [0., 0.], [-5., 2.]])
z = np.array([[-5., 2.], [0., -6.]])
assignment = np.zeros(x.shape[0])

assignment = assign_points(x, z)
print(assignment)
z = find_centroids(x, z, assignment)
print(z)
assignment = assign_points(x, z)
print(assignment)
z = find_centroids(x, z, assignment)
print(z)
assignment = assign_points(x, z)
print(assignment)
z = find_centroids(x, z, assignment)
print(z)
