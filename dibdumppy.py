from PIL import Image
import pickle
import sys

INPUT_DIR = 'images/'
OUTPUT_DIR = 'images_cleaned/'
TRAINING_DIR = 'training/'
TESTING_DIR = 'testing/'

CROP_WIDTH = 700
CROP_HEIGHT = 600
RM_TOP = 200


CROP = True
SAVE_RAW = True
DOWNSAMPLER = 12

def readBytes(n, file):
    return int.from_bytes(file.read(n), "little")

def cleanFile(imageName):
# !! static int bmpFileHeader_bfType;          // WORD
#    static int bmpFileHeader_bfSize;          // DWORD
#    static int bmpFileHeader_bfReserved1;     // WORD
#    static int bmpFileHeader_bfReserved2;     // WORD
#    static int bmpFileHeader_bfOffBits;       // DWORD

# !! static int bmpInfoHeader_biSize;          // DWORD
# !! static int bmpInfoHeader_biWidth;         // LONG
# !! static int bmpInfoHeader_biHeight;        // LONG
#    static int bmpInfoHeader_biPlanes;        // WORD
# !! static int bmpInfoHeader_biBitCount;      // WORD
#    static int bmpInfoHeader_biCompression;   // DWORD
#    static int bmpInfoHeader_biSizeImage;     // DWORD
#    static int bmpInfoHeader_biXPelsPerMeter; // LONG
#    static int bmpInfoHeader_biYPelsPerMeter; // LONG
# !! static int bmpInfoHeader_biClrUsed;       // DWORD
#    static int bmpInfoHeader_biClrImportant;  // DWORD

    # file header

    byteArr = []

    file = open(INPUT_DIR + imageName, "rb")

    if readBytes(2, file) != 19778:
        raise Exception("Type is not BM.")

    fileSize = readBytes(4, file)

    readBytes(8, file) # skip 8 bytes

    # info header

    headerSize = readBytes(4, file)
    if headerSize != 40:
        pass
        # print("Info Header is not 40 bytes.")
    
    width = readBytes(4, file) # 4
    height = readBytes(4, file) # 4

    readBytes(2, file) # skip 2 bytes

    bitCount = readBytes(2, file)

    readBytes(16, file) # skip 16 bytes

    clrUsed = readBytes(4, file)
    readBytes(4, file)

    # print(fileSize)
    # print(width)
    # print(height)
    # print(bitCount)
    # print(clrUsed)

    topDownDIB = height < 0
    height = -height if topDownDIB else height

    nColors = clrUsed
    match bitCount:
        case 1:
            nColors = 2
        case 2:
            nColors = 4
        case 4:
            nColors = 16
        case 8:
            nColors = 256

    readBytes(headerSize - 40, file)


    colorPallet = [[readBytes(1, file)] for i in range(nColors)] 

    pels = [[[] for j in range(width)] for i in range(height)]

    if bitCount == 32:
        for i in range(height):
            row = i if topDownDIB else height - 1 - i
            for j in range(width):
                pels[row][j] = (readBytes(1, file), readBytes(1, file), readBytes(1, file))
                readBytes(1, file) # dropped byte
    elif bitCount == 24:
        for i in range(height):
            row = i if topDownDIB else height - 1 - i
            for j in range(width):
                pels[row][j] = (readBytes(1, file), readBytes(1, file), readBytes(1, file))
        readBytes((4 - (width * 3) % 4) % 4, file)
    else:
        raise Exception("Bit count is not valid.")


    # crop out top RM_TOP rows of pels
    pels = pels[RM_TOP:]
    height -= RM_TOP

    if CROP:
        # grayscale, normalize, and flip pels
        gs_pels = [[1.0 - (pels[i][j][0] / 255.0) for j in range(width)] for i in range(height)]

        # calculate center of mass (com)
        sum_col = 0
        sum_row = 0
        sum_ovr = 0
        pel_val = 0

        for pels_row in range(height):
            for pels_col in range(width):
                pel_val = gs_pels[pels_row][pels_col]
                sum_col += pel_val * pels_col
                sum_row += pel_val * pels_row
                sum_ovr += pel_val

        com_col = int(sum_col / sum_ovr)
        com_row = int(sum_row / sum_ovr)

        com_col = 720
        com_row = 1580


        starting_col = com_col - int(CROP_WIDTH / 2.0)
        starting_row = com_row - int(CROP_HEIGHT / 2.0)

        pels = [[ ([255 - pels[starting_row + i][starting_col + j][0] for k in range(3)]) for j in range(CROP_WIDTH)] for i in range(CROP_HEIGHT)]

        width = CROP_WIDTH
        height = CROP_HEIGHT

    # downsample

    ds_pels = pels[0::DOWNSAMPLER]

    for i in range(len(ds_pels)):
        ds_pels[i] = ds_pels[i][0::DOWNSAMPLER]

    # save raw
    save_dir = TESTING_DIR if int(imageName.split('.')[0][1]) == 6 else TRAINING_DIR

    raw_pels = [[ds_pels[i][j][0] for j in range(len(ds_pels[0]))] for i in range(len(ds_pels))]
    print(str(len(raw_pels[0])) + " x " + str(len(raw_pels)))
    print(save_dir + imageName.split('.')[0] + '.in')
    raw_file = open(save_dir + imageName.split('.')[0] + '.in', "w")
    raw_file.write(("\n").join([("\n").join([str(1.0 - (raw_pel / 255.0)) for raw_pel in raw_pels_row]) for raw_pels_row in raw_pels]))
    raw_file.close()
        
    temp = []

    for row in ds_pels:
        for p in row:
            temp.append((p[2], p[1], p[0]))

    img = Image.new('RGB',(len(ds_pels[0]), len(ds_pels)))
    img.putdata(temp)
    img.save(OUTPUT_DIR + imageName)

# img = Image.new('RGB',(40, 40))
# test_arr = [(0, 0, 0) for i in range(40 * 40)]
# test_arr[10 * 40 + 10] = (255, 255, 255)
# img.putdata(test_arr)
# img.save('1a.bmp')

if len(sys.argv) != 2:
    print("Need to specify class of images.")

i = sys.argv[1]

for j in range(1, 7):
    cleanFile(str(i) + str(j) + '.bmp')
    print(str(i) + str(j) + '.bmp')