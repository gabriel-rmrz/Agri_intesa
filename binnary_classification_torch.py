DEBUG = True
import imageio.v3 as iio
import numpy as np
import torch
from torch.nn.functional import pad

def custom_collate_fn(batch):
  images, masks = zip(*batch)
  
  max_height = max(img.shape[1] for img in images)
  max_width = max(img.shape[2] for img in images)
  print(max_height)
  print(max_width)

  padded_images = []
  padded_masks = []

  for im, mask in zip(images, masks):
    pad_height = max_height - im.shape[1]
    pad_width = max_width - im.shape[2]

    img_padded = pad(im, (0, pad_width, 0, pad_height), value=0)
    mask_padded = pad(mask, (0, pad_width, 0, pad_height), value=0)
    print(img_padded.shape)
    print(mask_padded.shape)

    padded_images.append(img_padded)
    padded_masks.append(mask_padded)

  return torch.stack(padded_images), torch.stack(padded_masks)


def main():
  device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  input_files = ['Stefano_del_Sole_1_1_0',
                      'Atripalda_4_89_0']
  batch = []
  for in_file in input_files:
      mask_name = f"samples/array_mask_label_region_prov_{in_file}.npz" 
      data_file_name = f"samples/label_region_prov_{in_file}.png"
      data = iio.imread(data_file_name)
      data = np.tranpose(data, (2,0,1))
      data = torch.from_numpy(data)
      mask = np.load(mask_name)['arr_0']
      mask = mask.reshape((1, mask.shape[0], mask.shape[1]))
      mask = torch.from_numpy(mask)
      batch.append((data,mask))

  if DEBUG:
    print(f"Using device: {device}")
  batch = [(data, data_labels), (data_2, data_labels_2)]
  padded = custom_collate_fn(batch)
  del batch
  del padded
  del data
  del data_labels
  del data_2
  del data_labels_2

if __name__=='__main__':
  main()
