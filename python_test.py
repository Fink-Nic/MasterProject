# type: ignore
import momtrop
import random
import numpy


edge_1 = momtrop.Edge((0, 1), False, 0.66)
edge_2 = momtrop.Edge((1, 2), False, 0.77)
edge_3 = momtrop.Edge((2, 0), False, 0.88)


graph = momtrop.Graph([edge_1, edge_2, edge_3], [0, 1, 2])
signature = [[1], [1], [1]]

sampler = momtrop.Sampler(graph, signature)
edge_data = momtrop.EdgeData([0.0, 0.0, 0.0], [momtrop.Vector(
    0.0, 0.0, 0.0), momtrop.Vector(3., 4., 5.), momtrop.Vector(9., 11., 13.)])


test_point = [0.2, 0.3, 0.4, 0.423, 0.2324, 0.53, 0.9]

settings = momtrop.Settings(False, False)

sample_point = sampler.sample_point(
    test_point, edge_data, settings, force_sector=[1, 2, 0])

print("jacobian at test point: {}".format(sample_point.jacobian))
print("loop momenta at test point: {}".format(sample_point.loop_momenta))


subgraph_pdf = sampler.get_subgraph_pdf([0, 1, 2])
print("Distribution of edges on the full graph: {}".format(subgraph_pdf))

n_points = 100000

result_total = []

for _ in range(n_points):
    point = [random.random() for _ in range(9)]
    result_total.append([sampler.sample_point(
        point, edge_data, settings).jacobian])

numpy_res = numpy.array(result_total)

total_avg = numpy.average(numpy_res)
total_err = numpy.std(numpy_res) / numpy.sqrt(n_points)


print("integration result: {} +- {}".format(total_avg, total_err))
