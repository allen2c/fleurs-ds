import logging
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

if TYPE_CHECKING:
    from datasets import Dataset


logger = logging.getLogger(__name__)

HF_DATASETS_PARQUET_PATTERN = (
    "hf://datasets/google/fleurs@refs/convert/parquet/{lang}/{split}/*.parquet"
)


def get_dataset(
    name: Literal["google/fleurs"] = "google/fleurs",
    *,
    language: "LANGUAGE_TYPES",
    split: Literal["train", "dev", "test"],
) -> "Dataset":
    from datasets import DatasetDict

    from fleurs_ds import FLEURS_DATASETS_CACHE
    from fleurs_ds.types.features import features

    if language not in ALL_LANGUAGES:
        raise ValueError(
            f"Invalid language: {language}, must be one of {ALL_LANGUAGES}"
        )
    if split not in ["train", "dev", "test"]:
        raise ValueError(
            f"Invalid split: {split}, must be one of ['train', 'dev', 'test']"
        )

    cache_path = FLEURS_DATASETS_CACHE.joinpath(language)

    if cache_path.exists() and cache_path.is_dir() and any(cache_path.glob("*.json")):
        return DatasetDict.load_from_disk(str(cache_path))[split]

    train_dataset = _get_dataset_of_language_and_split(language, "train")
    dev_dataset = _get_dataset_of_language_and_split(language, "dev")
    test_dataset = _get_dataset_of_language_and_split(language, "test")

    dataset_dict = DatasetDict(
        {
            "train": train_dataset.cast(features),
            "dev": dev_dataset.cast(features),
            "test": test_dataset.cast(features),
        }
    )

    dataset_dict.save_to_disk(str(cache_path))
    return DatasetDict.load_from_disk(str(cache_path))[split]


def _get_dataset_of_language_and_split(
    language: "LANGUAGE_TYPES",
    split: Literal["train", "dev", "test"],
) -> "Dataset":
    from datasets import Audio, load_dataset

    valid_parquet_split: Literal["train", "validation", "test"] = (
        "validation" if split == "dev" else split
    )

    # Construct data files path
    data_files: dict[str, str] = {
        split: HF_DATASETS_PARQUET_PATTERN.format(
            lang=language, split=valid_parquet_split
        )
    }

    # Load dataset
    ds = load_dataset("parquet", data_files=data_files, split=split, streaming=False)

    # Rename transcription to text
    ds = ds.select_columns(["audio", "transcription"]).rename_column(
        "transcription", "text"
    )

    # Cast audio column to Audio
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))

    ds = ds.add_column("language", [language] * len(ds))

    # Select final columns
    ds = ds.select_columns(["audio", "text", "language"])

    # Cast to mp3 128k
    logger.info(f"Starting conversion for {language}:{split}...")
    ds = ds.map(
        _convert_audio_to_mp3_128k,
        num_proc=8,
        keep_in_memory=True,
        desc=f"Converting audio to mp3 128k for {language}:{split}",
    )

    original_count = len(ds)
    ds = ds.filter(
        lambda x: x["is_valid"],
        desc=f"Filtering corrupted samples for {language}:{split}",
    )
    filtered_count = len(ds)
    if filtered_count < original_count:
        logger.warning(
            f"Dropped {original_count - filtered_count} corrupted samples "
            + f"from {language}:{split}"
        )
    ds = ds.remove_columns(["is_valid"])

    return ds


def _convert_audio_to_mp3_128k(sample: dict) -> dict:
    import io
    import tempfile

    import soundfile as sf
    from pydub import AudioSegment

    error_result = {"audio": None, "is_valid": False}

    with tempfile.TemporaryDirectory() as _temp_dir:
        temp_dir = Path(_temp_dir)
        wav_path = temp_dir / "audio.wav"

        try:
            audio_array = sample["audio"]["array"]
        except Exception as e:
            logger.exception(e)
            logger.error(f"Error getting audio array for sample: {sample}")
            return error_result

        sampling_rate = sample["audio"]["sampling_rate"]

        sf.write(wav_path, audio_array, sampling_rate)
        # Audio processing logic (same as original)
        audio_seg: AudioSegment = AudioSegment.from_file(wav_path)
        audio_seg = audio_seg.set_channels(1).set_frame_rate(16000)
        duration = audio_seg.duration_seconds

        mp3_io = io.BytesIO()
        audio_seg.export(mp3_io, format="mp3", bitrate="128k")
        audio_bytes = mp3_io.getvalue()

        # Return the processed audio bytes
        return {"audio": audio_bytes, "is_valid": True, "duration": duration}


LANGUAGE_TYPES: TypeAlias = Literal[
    "af_za",
    "am_et",
    "ar_eg",
    "as_in",
    "ast_es",
    "az_az",
    "be_by",
    "bg_bg",
    "bn_in",
    "bs_ba",
    "ca_es",
    "ceb_ph",
    "ckb_iq",
    "cmn_hans_cn",
    "cs_cz",
    "cy_gb",
    "da_dk",
    "de_de",
    "el_gr",
    "en_us",
    "es_419",
    "et_ee",
    "fa_ir",
    "ff_sn",
    "fi_fi",
    "fil_ph",
    "fr_fr",
    "ga_ie",
    "gl_es",
    "gu_in",
    "ha_ng",
    "he_il",
    "hi_in",
    "hr_hr",
    "hu_hu",
    "hy_am",
    "id_id",
    "ig_ng",
    "is_is",
    "it_it",
    "ja_jp",
    "jv_id",
    "ka_ge",
    "kam_ke",
    "kea_cv",
    "kk_kz",
    "km_kh",
    "kn_in",
    "ko_kr",
    "ky_kg",
    "lb_lu",
    "lg_ug",
    "ln_cd",
    "lo_la",
    "lt_lt",
    "luo_ke",
    "lv_lv",
    "mi_nz",
    "mk_mk",
    "ml_in",
    "mn_mn",
    "mr_in",
    "ms_my",
    "mt_mt",
    "my_mm",
    "nb_no",
    "ne_np",
    "nl_nl",
    "nso_za",
    "ny_mw",
    "oc_fr",
    "om_et",
    "or_in",
    "pa_in",
    "pl_pl",
    "ps_af",
    "pt_br",
    "ro_ro",
    "ru_ru",
    "sd_in",
    "sk_sk",
    "sl_si",
    "sn_zw",
    "so_so",
    "sr_rs",
    "sv_se",
    "sw_ke",
    "ta_in",
    "te_in",
    "tg_tj",
    "th_th",
    "tr_tr",
    "uk_ua",
    "umb_ao",
    "ur_pk",
    "uz_uz",
    "vi_vn",
    "wo_sn",
    "xh_za",
    "yo_ng",
    "yue_hant_hk",
    "zu_za",
]
ALL_LANGUAGES = (
    "af_za",
    "am_et",
    "ar_eg",
    "as_in",
    "ast_es",
    "az_az",
    "be_by",
    "bg_bg",
    "bn_in",
    "bs_ba",
    "ca_es",
    "ceb_ph",
    "ckb_iq",
    "cmn_hans_cn",
    "cs_cz",
    "cy_gb",
    "da_dk",
    "de_de",
    "el_gr",
    "en_us",
    "es_419",
    "et_ee",
    "fa_ir",
    "ff_sn",
    "fi_fi",
    "fil_ph",
    "fr_fr",
    "ga_ie",
    "gl_es",
    "gu_in",
    "ha_ng",
    "he_il",
    "hi_in",
    "hr_hr",
    "hu_hu",
    "hy_am",
    "id_id",
    "ig_ng",
    "is_is",
    "it_it",
    "ja_jp",
    "jv_id",
    "ka_ge",
    "kam_ke",
    "kea_cv",
    "kk_kz",
    "km_kh",
    "kn_in",
    "ko_kr",
    "ky_kg",
    "lb_lu",
    "lg_ug",
    "ln_cd",
    "lo_la",
    "lt_lt",
    "luo_ke",
    "lv_lv",
    "mi_nz",
    "mk_mk",
    "ml_in",
    "mn_mn",
    "mr_in",
    "ms_my",
    "mt_mt",
    "my_mm",
    "nb_no",
    "ne_np",
    "nl_nl",
    "nso_za",
    "ny_mw",
    "oc_fr",
    "om_et",
    "or_in",
    "pa_in",
    "pl_pl",
    "ps_af",
    "pt_br",
    "ro_ro",
    "ru_ru",
    "sd_in",
    "sk_sk",
    "sl_si",
    "sn_zw",
    "so_so",
    "sr_rs",
    "sv_se",
    "sw_ke",
    "ta_in",
    "te_in",
    "tg_tj",
    "th_th",
    "tr_tr",
    "uk_ua",
    "umb_ao",
    "ur_pk",
    "uz_uz",
    "vi_vn",
    "wo_sn",
    "xh_za",
    "yo_ng",
    "yue_hant_hk",
    "zu_za",
)
