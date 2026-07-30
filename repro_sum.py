from tensorgraph.types import Obj
A = Obj("A")
B = Obj("B")
C = A + B
print(f"Result: {C}")
print(f"Type: {type(C)}")
assert C.name == "+"
