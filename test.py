tensor = preprocess_for_cnn_numpy(img)
logits_batch = self._triton_predict(tensor)

probs_batch = np.stack([softmax(logits) for logits in logits_batch])
classes = np.argmax(probs_batch, axis=1)
confidences = probs_batch[np.arange(len(classes)), classes]

if len(classes) == 1:
    # Ảnh chữ nhật: chỉ có ảnh gốc, không cần vote.
    cls = int(classes[0])
    conf = float(confidences[0])

elif len(classes) == 3:
    # Batch theo thứ tự: ảnh gốc, crop dọc, crop ngang.
    vote_counts = np.bincount(classes, minlength=len(ROTATION_MAP))
    majority_class = int(np.argmax(vote_counts))

    if vote_counts[majority_class] >= 2:
        # Có ít nhất hai ảnh cùng dự đoán một hướng.
        cls = majority_class
        conf = float(confidences[classes == majority_class].mean())
    else:
        # Cả ba hướng khác nhau: dùng dự đoán của ảnh gốc.
        cls = int(classes[0])
        conf = float(confidences[0])

else:
    raise ValueError(
        f"Expected preprocessing batch size 1 or 3, got {len(classes)}"
    )