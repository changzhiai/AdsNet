from ase.io import read, write
import os

def main():
    traj_path = 'metals_with_adss.traj'
    print(f'Reading {traj_path}...')
    images = read(traj_path, ':')
    print(f'Original count: {len(images)}')

    if len(images) > 12969:
        deleted_image = images.pop(12969)
        print(f'Removed index 12969: {deleted_image.get_chemical_formula()}')
        print(f'New count: {len(images)}')
        
        temp_path = 'new_metals_with_adss.traj'
        print(f'Writing to temporary file {temp_path}...')
        write(temp_path, images)
        
        print('Replacing original file...')
        os.replace(temp_path, traj_path)
        print('Done!')
    else:
        print('Error: Index 12969 is out of bounds.')

if __name__ == "__main__":
    main()
