import time
import torch


class A:


    def main(self):
        try:
            start_time = time.time()
            grids = torch.empty(1, 13566, 2)
           
            for i in range(3):
                if i == 0:
                    grid = torch.full((1, 10336, 2), 0)
                    grids[:, :10336, :] = grid
                elif i == 1:
                    grid = torch.full((1, 2584, 2), 1)
                    grids[:, 10336:10336 + 2584, :] = grid
                else:
                    grid = torch.full((1, 646, 2), 2)
                    grids[:, 10336 + 2584:, :] = grid
                
            print("--- %s seconds ---" % (time.time() - start_time))
        except Exception as e:
            print(e)

a = A()
for i in range(10):
    # time.sleep(0.5)
    a.main()