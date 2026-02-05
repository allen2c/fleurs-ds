from datasets import Audio, Features, Value

features = Features(
    {
        "audio": Audio(sampling_rate=16000),
        "text": Value("string"),
        "language": Value("string"),
    }
)
