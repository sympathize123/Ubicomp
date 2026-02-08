from pytorch_widedeep.models import SAINT, TabTransformer
import inspect

with open('signatures.txt', 'w') as f:
    f.write(f"SAINT: {inspect.signature(SAINT.__init__)}\n")
    f.write(f"TabTransformer: {inspect.signature(TabTransformer.__init__)}\n")
