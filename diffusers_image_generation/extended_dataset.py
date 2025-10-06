import glob
from torch.utils.data import Dataset
from PIL import Image

class NestedImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        
        # Find all image files in subdirectories
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        self.image_paths = []
        
        for ext in image_extensions:
            # Search recursively for images
            pattern = os.path.join(root_dir, '**', ext)
            self.image_paths.extend(glob.glob(pattern, recursive=True))
            
            # Also search for uppercase extensions
            pattern = os.path.join(root_dir, '**', ext.upper())
            self.image_paths.extend(glob.glob(pattern, recursive=True))
        
        print(f"Found {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return {"input": image}