import xml.etree.ElementTree as etree


def convert(input_file, output_file):
    tree = etree.parse(input_file)
    root = tree.getroot()
    with open(output_file, 'w') as f:
        for review in root.findall('.//Review'):
            text = ""
            for sentence in review.findall('.//text'):
                text += sentence.text + " "
            f.write(text + '\n')

# convert("ABSA16_Restaurants_Ru_Train.xml", "train.txt")
# convert("ABSA16_Restaurants_Ru_Test.xml", "test.txt")
