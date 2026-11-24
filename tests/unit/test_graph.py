from app.agents.routing import RouteInput, route_request


def test_text_only_goes_to_txt2img() -> None:
    assert route_request(RouteInput(has_images=False)) == "txt2img"


def test_uploaded_image_goes_to_img2img() -> None:
    assert route_request(RouteInput(has_images=True)) == "img2img"
