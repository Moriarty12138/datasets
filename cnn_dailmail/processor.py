#!/usr/bin/env python
import os
import argparse
import hashlib
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)


DM_SINGLE_CLOSE_QUOTE = "\u2019"  # unicode
DM_DOUBLE_CLOSE_QUOTE = "\u201d"
# acceptable ways to end a sentence
END_TOKENS = [".", "!", "?", "...", "'", "`", '"', DM_SINGLE_CLOSE_QUOTE, DM_DOUBLE_CLOSE_QUOTE, ")"]


def _get_url_hashes(path):
    """Get hashes of urls in file."""
    urls = _read_text_file(path)

    def url_hash(u):
        h = hashlib.sha1()
        try:
            u = u.encode("utf-8")
        except UnicodeDecodeError:
            logger.error("Cannot hash url: %s", u)
        h.update(u)
        return h.hexdigest()

    return {url_hash(u): True for u in urls}


def _get_hash_from_path(p):
    """Extract hash from path."""
    basename = os.path.basename(p)
    return basename[0: basename.find(".story")]


def _find_files(dl_paths, publisher, url_dict):
    """Find files corresponding to urls.
    :parameter:
        dl_paths: {`cnn_stories`:} or {`dm_stories`:}
        publisher: `cnn` or `dm`
        url_dict: dict of data's path
    :returns
        ret_files: list of files
    """
    if publisher == "cnn":
        top_dir = os.path.join(dl_paths["cnn_stories"], "cnn", "stories")
    elif publisher == "dm":
        top_dir = os.path.join(dl_paths["dm_stories"], "dailymail", "stories")
    else:
        logger.fatal("Unsupported publisher: %s", publisher)
    files = sorted(os.listdir(top_dir))

    ret_files = []
    for p in files:
        if _get_hash_from_path(p) in url_dict:
            ret_files.append(os.path.join(top_dir, p))
    return ret_files


def _read_text_file(text_file):
    """read data from file."""
    lines = []
    with open(text_file, "r", encoding="utf-8") as f:
        for line in f:
            lines.append(line.strip())
    return lines


def _get_art_abs(story_file, tfds_version):
    """Get abstract (highlights) and article from a story file path."""
    # Based on https://github.com/abisee/cnn-dailymail/blob/master/
    #     make_datafiles.py

    lines = _read_text_file(story_file)

    # The github code lowercase the text and we removed it in 3.0.0.

    # Put periods on the ends of lines that are missing them
    # (this is a problem in the dataset because many image captions don't end in
    # periods; consequently they end up in the body of the article as run-on
    # sentences)
    def fix_missing_period(line):
        """Adds a period to a line that is missing a period."""
        if "@highlight" in line:
            return line
        if not line:
            return line
        if line[-1] in END_TOKENS:
            return line
        return line + " ."

    lines = [fix_missing_period(line) for line in lines]

    # Separate out article and abstract sentences
    article_lines = []
    highlights = []
    next_is_highlight = False
    for line in lines:
        if not line:
            continue  # empty line
        elif line.startswith("@highlight"):
            next_is_highlight = True
        elif next_is_highlight:
            highlights.append(line)
        else:
            article_lines.append(line)

    # Make article into a single string
    article = " ".join(article_lines)

    if tfds_version >= "2.0.0":
        abstract = "\n".join(highlights)
    else:
        abstract = " ".join(highlights)

    return article, abstract


def _subset_filenames(dl_paths, split):
    """Get filenames for a particular split."""
    assert isinstance(dl_paths, dict), dl_paths
    # Get filenames for a split.
    if split == 'train':
        urls = _get_url_hashes(dl_paths["train_urls"])
    elif split == 'val':
        urls = _get_url_hashes(dl_paths["val_urls"])
    elif split == 'test':
        urls = _get_url_hashes(dl_paths["test_urls"])
    else:
        logger.fatal("Unsupported split: %s", split)
    cnn = _find_files(dl_paths, "cnn", urls)
    dm = _find_files(dl_paths, "dm", urls)
    return cnn + dm


def save_data(filenames,
              src_path="cnn_dailymail/train.src",
              tgt_path="cnn_dailymail/train.tgt"):
    """save dataset into file."""
    srcs = []
    tgts = []
    for p in filenames:
        article, highlights = _get_art_abs(p, '3.0.0')
        if not article or not highlights:
            continue
        srcs.append(article)
        tgts.append(highlights)
    assert len(srcs) == len(tgts), "article should match to highlights."
    with open(src_path, 'w', encoding='utf-8') as writer:
        for src in tqdm(srcs, total=len(srcs)):
            writer.write(src.strip() + '\n')
    with open(tgt_path, 'w', encoding='utf-8') as writer:
        for tgt in tqdm(tgts, total=len(tgts)):
            tgt = tgt.replace('\n','<n>')  # 替换摘要中的换行符，使用数据的时候可以替换回来
            writer.write(tgt.strip() + '\n')


def main(args):
    # dl_paths
    dl_paths = {"train_urls": args.all_train,
                "test_urls": args.all_test,
                "val_urls": args.all_val,
                "cnn_stories": args.cnn_stories,
                "dm_stories": args.dm_stories}

    # filenames
    train_filenames = _subset_filenames(dl_paths, 'train')
    test_filenames = _subset_filenames(dl_paths, 'test')
    val_filenames = _subset_filenames(dl_paths, 'val')

    # save dataset
    save_data(train_filenames, args.train_src, args.train_tgt)
    save_data(test_filenames, args.test_src, args.test_tgt)
    save_data(val_filenames, args.val_src, args.val_tgt)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process cnn daliymail dataset.')
    # load dataset
    parser.add_argument('--all_train',
                        default="/datasets/cnn_dailymail/all_train.txt",
                        type=str,
                        help='path to all_train.txt. example: cnn_dailymail/all_train.txt')
    parser.add_argument('--all_test',
                        default="/datasets/cnn_dailymail/all_test.txt",
                        type=str,
                        help='path to all_test.txt. example: cnn_dailymail/all_test.txt')
    parser.add_argument('--all_val',
                        default="/datasets/cnn_dailymail/all_val.txt",
                        type=str,
                        help='path to all_val.txt. example: cnn_dailymail/all_val.txt')
    parser.add_argument('--cnn_stories',
                        default="/datasets/cnn_dailymail/cnn_stories",
                        type=str,
                        help='path to cnn_stories/. example: cnn_dailymail/cnn_stories')
    parser.add_argument('--dm_stories',
                        default="/datasets/cnn_dailymail/dailymail_stories",
                        type=str,
                        help='path to dailymail_storie/. example: cnn_dailymail/dailymail_storie')

    # save dataset
    parser.add_argument('--train_src',
                        default="cnn_dailymail/train.src",
                        type=str,
                        help='path to save train.src.')
    parser.add_argument('--train_tgt',
                        default="cnn_dailymail/train.tgt",
                        type=str,
                        help='path to save train.tgt.')
    parser.add_argument('--test_src',
                        default="cnn_dailymail/test.src",
                        type=str,
                        help='path to save test.src.')
    parser.add_argument('--test_tgt',
                        default="cnn_dailymail/test.tgt",
                        type=str,
                        help='path to save test.tgt.')
    parser.add_argument('--val_src',
                        default="cnn_dailymail/val.src",
                        type=str,
                        help='path to save val.src.')
    parser.add_argument('--val_tgt',
                        default="cnn_dailymail/val.tgt",
                        type=str,
                        help='path to save val.tgt.')

    args = parser.parse_args()

    main(args)
