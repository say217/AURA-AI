# for testing modules and its use

import subprocess

import pyttsx3

_sapi_speaker = None


def _pick_female_voice(engine: pyttsx3.Engine) -> str | None:
	voices = engine.getProperty("voices")
	for voice in voices:
		name = (voice.name or "").lower()
		gender = (getattr(voice, "gender", "") or "").lower()
		if "female" in name or "female" in gender or "zira" in name:
			return voice.id
	return None


def _try_init_sapi_speaker() -> bool:
	global _sapi_speaker
	if _sapi_speaker is not None:
		return True
	try:
		import win32com.client  # type: ignore

		_sapi_speaker = win32com.client.Dispatch("SAPI.SpVoice")
		for voice in _sapi_speaker.GetVoices():
			desc = (voice.GetDescription() or "").lower()
			if "female" in desc or "zira" in desc:
				_sapi_speaker.Voice = voice
				break
		return True
	except Exception:
		_sapi_speaker = None
		return False


def _speak_with_powershell(text: str) -> None:
	escaped = text.replace('"', '`"')
	command = (
		"Add-Type -AssemblyName System.Speech; "
		"$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
		"$s.SelectVoiceByHints('Female'); "
		f"$s.Speak(\"{escaped}\")"
	)
	subprocess.run(
		["powershell", "-NoProfile", "-Command", command],
		check=False,
		capture_output=True,
		text=True,
	)


def speak_text(text: str) -> None:
	if _try_init_sapi_speaker():
		_sapi_speaker.Speak(text)
		return

	try:
		engine = pyttsx3.init("sapi5")
		voice_id = _pick_female_voice(engine)
		if voice_id:
			engine.setProperty("voice", voice_id)
		engine.say(text)
		engine.runAndWait()
		engine.stop()
	except Exception:
		_speak_with_powershell(text)


def speak_loop() -> None:
	print("Type something and press Enter. Type 'exit' to quit.")
	while True:
		text = input("> ").strip()
		if not text:
			continue
		if text.lower() in {"exit", "quit"}:
			break

		speak_text(text)


if __name__ == "__main__":
	speak_loop()


