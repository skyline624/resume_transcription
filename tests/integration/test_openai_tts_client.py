from openai import OpenAI


def test_sdk_openai_ecrit_un_wav_depuis_api_live(live_tts_api, tmp_path):
    output = tmp_path / "speech.wav"
    with OpenAI(api_key="local", base_url=live_tts_api.base_url + "/v1") as client:
        with client.audio.speech.with_streaming_response.create(
            model="tts-1-hd",
            voice="Ryan",
            input="Bonjour depuis le client OpenAI.",
            response_format="wav",
        ) as speech:
            speech.stream_to_file(output)

    assert output.read_bytes().startswith(b"RIFF")
