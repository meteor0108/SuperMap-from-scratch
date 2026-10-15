"""VLM interface, answer parser, and waypoint conversion (REQ-057..REQ-059, Sec. IV-D, V-F)."""
# REQ-057..REQ-059: public API
from supermap.vlm.backend import GeminiBackend, VLMBackend, find_api_key, make_backend
from supermap.vlm.grounding import GroundingResult, build_prompt, ground, run_grounding
from supermap.vlm.parser import parse_answer
from supermap.vlm.waypoints import ids_to_waypoints
