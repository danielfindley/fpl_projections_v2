from unittest.mock import Mock

import scrape_update_data as scraper


def test_fotmob_browser_falls_back_to_direct_api_when_chrome_disconnects(monkeypatch):
    driver = Mock()
    driver.get_cookies.side_effect = ConnectionError("Chrome session closed")
    monkeypatch.setattr(scraper.uc, "Chrome", lambda **kwargs: driver)

    response = Mock(status_code=200)
    response.json.return_value = {"fixtures": {"allMatches": []}}
    session = Mock()
    session.headers = {}
    session.get.return_value = response
    monkeypatch.setattr(scraper.requests, "Session", lambda: session)

    with scraper.FotMobBrowser() as browser:
        assert browser.fetch_json("/api/data/leagues?id=47") == response.json.return_value

    assert session.headers["User-Agent"] == scraper.HEADERS["User-Agent"]
    driver.quit.assert_called_once()
