import sys
sys.path.append(r'c:\Users\koula\Desktop\inferno')
from inferno.core.model_manager import ModelManager
print('Imported ModelManager OK')
mm = ModelManager()
print(mm.parse_model_string('hf:repo:Q4_K_M'))
print(mm.parse_model_string('repo:filename.gguf'))
