import argparse

import pandas as pd
from torch.utils.data import DataLoader
from models import *
from dataloader2 import *
from utils import *
import gdown
def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    # pixels_greater_than_255 = pixels[pixels > 255]
    # print("Pixel values greater than 255:", pixels_greater_than_255)
    pixels[pixels >= 225] = 255
    pixels[pixels < 225] = 0
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded

    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)

def mask2string(dir):
    ## mask --> string
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        img = cv2.imread(path)[:,:,::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:,:,channel])
            strings.append(string)
    r = {
        'ids': ids,
        'strings': strings,
    }
    return r

def parse_arguments():
    parser = argparse.ArgumentParser(description='Hello')

    # Add arguments
    parser.add_argument('--epochs', type=int, default=2, help='Number of testing epochs')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoint/model_end.pth', help='Path to checkpoint file')
    parser.add_argument('--test_image_path', type=str, default='dataset/test/test', help='Path to test image file')
    parser.add_argument('--predicted_path', type=str, default='predicted_masks', help='Predicted mask folder')

    args = parser.parse_args()

    return args

def main():
    config = vars(parse_arguments())

    CHECKPOINT_FILE = config["checkpoint_path"]

    print(f"Checkpoint path for validation: {CHECKPOINT_FILE}, check if file exists: {os.path.exists(CHECKPOINT_FILE)}")

    IMAGE_TESTING_PATH = config["test_image_path"]
    TEST_BATCH_SIZE = config["epochs"]
    PREDICTED_MASK_PATH = config["predicted_path"]

    loaded_checkpoint = torch.load(CHECKPOINT_FILE)

    last_epoch = loaded_checkpoint['epoch'] if 'epoch' in loaded_checkpoint else -1
    num_classes = loaded_checkpoint['num_classes'] if 'num_classes' in loaded_checkpoint else 3
    inp_channels = loaded_checkpoint['input_channels'] if 'input_channels' in loaded_checkpoint else 3

    model = UNet(num_classes, inp_channels).to(DEVICE)

    model.load_state_dict(loaded_checkpoint['model'])
    model.eval()
    model = model.to('cpu')

    all_dataset = SplitDataset(testing_images_path=IMAGE_TESTING_PATH)
    test_set = all_dataset.create_testing_set()

    test_dataloader = DataLoader(test_set, batch_size=TEST_BATCH_SIZE, shuffle=True)

    if not os.path.isdir(PREDICTED_MASK_PATH):
        os.mkdir(PREDICTED_MASK_PATH)
    for _, (img, path, H, W) in enumerate(test_dataloader):
        a = path
        b = img
        h = H
        w = W

        with torch.no_grad():
            predicted_mask = model(b)

        for i in range(len(a)):
            pred_mask = predicted_mask[i].squeeze(0).cpu().numpy().transpose(1, 2, 0)
            image_id = a[i].split('/')[-1].split('.')[0]
            filename = image_id + ".png"
            mask_path = os.path.join(PREDICTED_MASK_PATH, filename)

            mask = cv2.resize(pred_mask, (h[i].item(), w[i].item()))
            mask = np.argmax(mask, axis=2)
            mask_rgb = mask_to_rgb(mask)
            mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(mask_path, mask_rgb)

    res = mask2string(PREDICTED_MASK_PATH)
    df = pd.DataFrame(columns=['Id', 'Expected'])
    df['Id'] = res['ids']
    df['Expected'] = res['strings']
    df.to_csv(r'output.csv', index=False)

if __name__ == "__main__":
    main()