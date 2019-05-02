import torch
import torch.nn as nn
from torch.autograd import Variable

from modules.nnd import NNDModule

dist =  NNDModule()

p1 = torch.rand(100,5000,3)*20
p2 = torch.rand(100,5000,3)*20
# p1 = p1.int()
# p1.random_(0,2)
# p1 = p1.float()
# p2 = p2.int()
# p2.random_(0,2)
p2 = p2.float()
# print(p1)
# print(p2)


for i in range(100000):
    print('gpu')
    points1_cuda = Variable(p1.cuda(), requires_grad = True)
    points2_cuda = Variable(p2.cuda())
    dist1_cuda, dist2_cuda = dist(points1_cuda, points2_cuda)
    print(dist1_cuda, dist2_cuda)
    loss_cuda = torch.sum(dist1_cuda)
    print(loss_cuda)
    loss_cuda.backward()
    print(points1_cuda.grad, points2_cuda.grad)

    print('stats:')
    print('loss :', loss_cuda)
