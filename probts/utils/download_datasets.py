import gdown
import shutil
import os
import argparse

def download_and_extract_zip(output_path, zip_name='all_datasets'):
    gdown.download(id='1tSc1WA30CL2aMt5hAW7M-d5_0IBz-lJP', output=output_path, quiet=False)
    print(f"Data files are saved to {os.path.dirname(output_path)}")

    file_path = output_path + zip_name + '.zip'
    
    try:
        shutil.unpack_archive(file_path, os.path.dirname(file_path))
        print(f"files are unzipped")
    except shutil.ReadError:
        print("is not zip file")
        
    move_files_up_one_level(output_path + zip_name)
    cleanup_directory(output_path)
    print("datasets prepared done.")
    
def move_files_up_one_level(directory):
    for item in os.listdir(directory):
        s = os.path.join(directory, item)
        d = os.path.join(os.path.dirname(directory), item)
        shutil.move(s, d)
    os.rmdir(directory)
    
def cleanup_directory(directory):
    for root, dirs, files in os.walk(directory):
        for name in dirs:
            if name in ['__MACOSX']:
                shutil.rmtree(os.path.join(root, name))
                
        for name in files:
            if name in ['.DS_Store', 'all_datasets.zip']:
                os.remove(os.path.join(root, name))
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download and extract zip file from Google Drive')
    parser.add_argument('--data_path', type=str, required=True, help='Path to store the extracted files')
    args = parser.parse_args()

    download_and_extract_zip(args.data_path, zip_name='all_datasets')