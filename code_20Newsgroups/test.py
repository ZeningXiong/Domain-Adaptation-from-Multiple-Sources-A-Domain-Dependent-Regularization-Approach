import os

def check_file():
    file_path = r"f:\港大sem2\COMP7404\group project\7404_p\7404_p\code_20Newsgroups\temp\svm_data_round0.mat"
    if os.path.exists(file_path):
        print(f"文件存在: {file_path}")
        print(f"文件大小: {os.path.getsize(file_path)} 字节")
    else:
        print(f"文件不存在: {file_path}")
        
        # 检查temp目录是否存在
        temp_dir = os.path.dirname(file_path)
        if os.path.exists(temp_dir):
            print(f"temp目录存在，内容:")
            for item in os.listdir(temp_dir):
                print(f"- {item}")
        else:
            print(f"temp目录不存在: {temp_dir}")

if __name__ == "__main__":
    check_file()