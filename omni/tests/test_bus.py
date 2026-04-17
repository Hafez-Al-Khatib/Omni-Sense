"""Event bus: publish/subscribe, fan-out isolation, stop/start."""
import asyncio

import pytest

from omni.common.bus import InMemoryBus


@pytest.mark.asyncio
async def test_single_subscriber_receives_message():
    bus = InMemoryBus()
    received = []

    async def handler(payload):
        received.append(payload)

    bus.subscribe("test.topic", handler)
    task = asyncio.create_task(bus.run())

    await bus.publish("test.topic", {"value": 42})
    await asyncio.sleep(0.05)
    bus.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert len(received) == 1
    assert received[0]["value"] == 42


@pytest.mark.asyncio
async def test_fan_out_multiple_subscribers():
    bus = InMemoryBus()
    log_a, log_b = [], []

    async def a(p): log_a.append(p)
    async def b(p): log_b.append(p)

    bus.subscribe("x", a)
    bus.subscribe("x", b)
    task = asyncio.create_task(bus.run())

    await bus.publish("x", {"n": 1})
    await asyncio.sleep(0.05)
    bus.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert log_a == [{"n": 1}]
    assert log_b == [{"n": 1}]


@pytest.mark.asyncio
async def test_failing_handler_does_not_poison_others():
    bus = InMemoryBus()
    ok_log = []

    async def bad(p):
        raise RuntimeError("intentional failure")

    async def good(p):
        ok_log.append(p)

    bus.subscribe("evt", bad)
    bus.subscribe("evt", good)
    task = asyncio.create_task(bus.run())

    await bus.publish("evt", {"msg": "hello"})
    await asyncio.sleep(0.05)
    bus.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert ok_log == [{"msg": "hello"}]


@pytest.mark.asyncio
async def test_no_subscribers_no_crash():
    bus = InMemoryBus()
    task = asyncio.create_task(bus.run())
    await bus.publish("orphan.topic", {"x": 1})
    await asyncio.sleep(0.02)
    bus.stop()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
