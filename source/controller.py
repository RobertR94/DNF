from dnf import DNF

class DNFController:

    def run(self):
        dnf = DNF()
        theta = dnf.train()
        print(theta)
