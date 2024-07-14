import os
# folder_path = r'..\Q&A_System\test\vector_store'
folder_path = r'.\vector_store\test'
contents = os.listdir(folder_path)
subfolders = [f for f in contents]
print(subfolders)