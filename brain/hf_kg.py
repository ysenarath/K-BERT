def add_knowledge_with_vm(
    tokenizer,
    lookup_table: dict,
    sent_batch,
    special_tags,
    max_entities=2,
    max_length=128,
):
    """
    input: sent_batch - list of sentences, e.g., ["abcd", "efgh"]
    return: know_sent_batch - list of sentences with entites embedding
            position_batch - list of position index of each character.
            visible_matrix_batch - list of visible matrixs
            seg_batch - list of segment tags
    """
    split_sent_batch = [tokenizer.cut(sent) for sent in sent_batch]
    know_sent_batch = []
    position_batch = []
    visible_matrix_batch = []
    seg_batch = []
    for split_sent in split_sent_batch:
        # create tree
        sent_tree = []
        pos_idx_tree = []
        abs_idx_tree = []
        pos_idx = -1
        abs_idx = -1
        abs_idx_src = []
        for token in split_sent:
            entities = list(lookup_table.get(token, []))[:max_entities]
            sent_tree.append((token, entities))

            if token in special_tags:
                token_pos_idx = [pos_idx + 1]
                token_abs_idx = [abs_idx + 1]
            else:
                token_pos_idx = [pos_idx + i for i in range(1, len(token) + 1)]
                token_abs_idx = [abs_idx + i for i in range(1, len(token) + 1)]
            abs_idx = token_abs_idx[-1]

            entities_pos_idx = []
            entities_abs_idx = []
            for ent in entities:
                ent_pos_idx = [token_pos_idx[-1] + i for i in range(1, len(ent) + 1)]
                entities_pos_idx.append(ent_pos_idx)
                ent_abs_idx = [abs_idx + i for i in range(1, len(ent) + 1)]
                abs_idx = ent_abs_idx[-1]
                entities_abs_idx.append(ent_abs_idx)

            pos_idx_tree.append((token_pos_idx, entities_pos_idx))
            pos_idx = token_pos_idx[-1]
            abs_idx_tree.append((token_abs_idx, entities_abs_idx))
            abs_idx_src += token_abs_idx

        # Get know_sent and pos
        know_sent = []
        pos = []
        seg = []
        for i in range(len(sent_tree)):
            word = sent_tree[i][0]
            if word in self.special_tags:
                know_sent += [word]
                seg += [0]
            else:
                add_word = list(word)
                know_sent += add_word
                seg += [0] * len(add_word)
            pos += pos_idx_tree[i][0]
            for j in range(len(sent_tree[i][1])):
                add_word = list(sent_tree[i][1][j])
                know_sent += add_word
                seg += [1] * len(add_word)
                pos += list(pos_idx_tree[i][1][j])

        token_num = len(know_sent)

        # Calculate visible matrix
        visible_matrix = np.zeros((token_num, token_num))
        for item in abs_idx_tree:
            src_ids = item[0]
            for id in src_ids:
                visible_abs_idx = abs_idx_src + [idx for ent in item[1] for idx in ent]
                visible_matrix[id, visible_abs_idx] = 1
            for ent in item[1]:
                for id in ent:
                    visible_abs_idx = ent + src_ids
                    visible_matrix[id, visible_abs_idx] = 1

        src_length = len(know_sent)
        if len(know_sent) < max_length:
            pad_num = max_length - src_length
            know_sent += [config.PAD_TOKEN] * pad_num
            seg += [0] * pad_num
            pos += [max_length - 1] * pad_num
            visible_matrix = np.pad(
                visible_matrix, ((0, pad_num), (0, pad_num)), "constant"
            )  # pad 0
        else:
            know_sent = know_sent[:max_length]
            seg = seg[:max_length]
            pos = pos[:max_length]
            visible_matrix = visible_matrix[:max_length, :max_length]

        know_sent_batch.append(know_sent)
        position_batch.append(pos)
        visible_matrix_batch.append(visible_matrix)
        seg_batch.append(seg)

    return know_sent_batch, position_batch, visible_matrix_batch, seg_batch
