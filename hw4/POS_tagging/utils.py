def load_tag_data(tag_file):
    all_sentences = []
    all_tags = []
    sent = []
    tags = []
    with open(tag_file, 'r') as f:
        for line in f:
            if line.strip() == "":
                all_sentences.append(sent)
                all_tags.append(tags)
                sent = []
                tags = []
            else:
                word, tag, _ = line.strip().split()
                sent.append(word)
                tags.append(tag)
    return all_sentences, all_tags

def load_txt_data(txt_file):
    all_sentences = []
    sent = []
    with open(txt_file, 'r') as f:
        for line in f:
            if(line.strip() == ""):
                all_sentences.append(sent)
                sent = []
            else:
                word = line.strip()
                sent.append(word)
    return all_sentences
