import tempfile

temp_dir = tempfile.TemporaryDirectory()
directory_path = temp_dir.name.replace("\\","/")+"/"