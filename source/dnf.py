import copy
import json

class DNF:

    def __init__(self):
        self.theta = None
        return
    

    def initialize(self, path_samples):
        with open(path_samples) as j:
            data = json.load(j)
        
        self.samples = data["samples"]
        self.attributes = data["attributes"]

        return
    

    def power_set_generator(self, source):
        for i in range(2**len(source)):
            elem = []
            for j in range(len(source)):
                if i & 1<<j:
                    elem.append(source[j])
            yield elem


    def calc_gamma(self, source):
        gamma = list()
        for e1 in source:
            for e2 in source:
                if set(e1).isdisjoint(set(e2)) and e1 != e2:
                    gamma.append((e1, e2))  
        return gamma


    def func(self, theta, sample):
        result = 0
        con = 1
        for (v0, v1) in theta:
            for v in v0:
                con = con * self.samples[sample][v]
            con = 1 - con
            for v in v1:
                con = con * self.samples[sample][v]
            result = result or con
        return result


    def reg_length(self, theta):
        length = 0
        for (v0, v1) in theta:
            length = len(v0) + len(v1)
        return length


    def loss_func(a, b):
        if a == b:
            return 0
        else:
            return 1


    def solve_instace(self, theta, delta):
        result = delta * self.reg_length(theta)
        sum_result = 0
        for s in self.samples.items():
            sum_result = sum_result + self.loss_func(self.func(theta, s), s["y"])
        result = result + (1/len(self.samples)) * sum_result
        return result


    def train(self):
        self.initialize("../data/samples.json")
        p_set = list()
        power_of_v = [x for x in self.power_set_generator(self.attributes)]
        gamma = self.calc_gamma(power_of_v)
        capital_theta = self.power_set_generator(gamma)
        min_theta = None
        delta = 0,8
        for theta in capital_theta:
            result = (theta, self.solve_instace(theta, delta))
            if result[0] < min_theta or min_theta == None:
                min_theta = copy.deepcopy(result)

        self.theta = copy.deepcopy(min_theta)
        return min_theta  