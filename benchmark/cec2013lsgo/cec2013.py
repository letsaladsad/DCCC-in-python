from .f1 import F1 as f1
from .f2 import F2 as f2
from .f3 import F3 as f3
from .f4 import F4 as f4
from .f5 import F5 as f5
from .f6 import F6 as f6
from .f7 import F7 as f7
from .f8 import F8 as f8
from .f9 import F9 as f9
from .f10 import F10 as f10
from .f11 import F11 as f11
from .f12 import F12 as f12
from .f13 import F13 as f13
from .f14 import F14 as f14
from .f15 import F15 as f15

class Benchmark():
    def get_function(self, func_id):
        if func_id == 1:
            return f1()
        elif func_id == 2:
            return f2()
        elif func_id == 3:
            return f3()
        elif func_id == 4:
            return f4()
        elif func_id == 5:
            return f5()
        elif func_id == 6:
            return f6()
        elif func_id == 7:
            return f7()
        elif func_id == 8:
            return f8()
        elif func_id == 9:
            return f9()
        elif func_id == 10:
            return f10()
        elif func_id == 11:
            return f11()
        elif func_id == 12:
            return f12()
        elif func_id == 13:
            return f13()
        elif func_id == 14:
             return f14()
        elif func_id == 15:
            return f15()
        else:
            raise ValueError("Function id is out of range.")

    def get_info(self, func_id):
        if func_id == 1:
            f1_ = f1()
            return f1_.info()
        elif func_id == 2:
            f2_ = f2()
            return f2_.info()
        elif func_id == 3:
            f3_ = f3()
            return f3_.info()
        elif func_id == 4:
            f4_ = f4()
            return f4_.info()
        elif func_id == 5:
            f5_ = f5()
            return f5_.info()
        elif func_id == 6:
            f6_ = f6()
            return f6_.info()
        elif func_id == 7:
            f7_ = f7()
            return f7_.info()
        elif func_id == 8:
            f8_ = f8()
            return f8_.info()
        elif func_id == 9:
            f9_ = f9()
            return f9_.info()
        elif func_id == 10:
            f10_ = f10()
            return f10_.info()
        elif func_id == 11:
            f11_ = f11()
            return f11_.info()
        elif func_id == 12:
            f12_ = f12()
            return f12_.info()
        elif func_id == 13:
            f13_ = f13()
            return f13_.info()
        elif func_id == 14:
            f14_ = f14()
            return f14_.info()
        elif func_id == 15:
            f15_ = f15()
            return f15_.info()
        else:
            raise ValueError("Function id is out of range.")
    
    def get_num_functions(self):
        return 15



