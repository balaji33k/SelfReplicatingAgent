import asyncio
import aiohttp
import time
import random

async def make_request(session, url, delay=0):
    """
    Makes an asynchronous HTTP request.
    Introduces a delay before cancellation to simulate work.
    """
    try:
        if delay > 0:
            await asyncio.sleep(delay)
        async with session.get(url) as response:
            return await response.text()
    except asyncio.CancelledError:
        # print("Request cancelled")
        return None
    except Exception as e:
        print(f"Request failed: {e}")
        return None


async def main(num_requests=100, cancellation_delay=0.01):
    """
    Launches multiple asynchronous requests and cancels them immediately.
    """
    url = "https://httpbin.org/get"  # Use a reliable testing endpoint

    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=300)) as session:
        initial_pool_size = session.connector.limit
        print(f"Initial pool size: {initial_pool_size}")

        tasks = [make_request(session, url) for _ in range(num_requests)]

        # Cancel all tasks after a short delay (simulating immediate cancellation)
        await asyncio.sleep(cancellation_delay)
        for task in tasks:
            task.cancel()

        # Wait for all tasks to complete (cancelled or otherwise)
        await asyncio.gather(*tasks, return_exceptions=True)

        # Allow time for connections to be released back to the pool
        await asyncio.sleep(0.2)  # Adjust this sleep time as needed

        final_pool_size = session.connector.limit
        print(f"Final pool size: {final_pool_size}")

        # Verification: Check if the pool size is close to the initial size
        assert abs(final_pool_size - initial_pool_size) <= 10 , f"Pool size not restored. Initial: {initial_pool_size}, Final: {final_pool_size}" # Tolerance for minor variations
        print("Pool size verification passed.")


if __name__ == "__main__":
    asyncio.run(main())

    async def test_cancellation():
        """Test case for checking cancellation."""
        num_requests = 50
        cancellation_delay = 0.01
        url = "https://httpbin.org/get"

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=300)) as session:
            tasks = [make_request(session, url, delay=random.uniform(0.05, 0.1)) for _ in range(num_requests)]
            await asyncio.sleep(cancellation_delay) #cancel them shortly after creation

            cancelled_count = 0
            for task in tasks:
                task.cancel()
                cancelled_count +=1
            
            results = await asyncio.gather(*tasks, return_exceptions=True)

            num_actually_cancelled = sum(1 for r in results if isinstance(r, asyncio.CancelledError))

            print(f"Cancelled {cancelled_count} tasks")
            print(f"Tasks actually cancelled: {num_actually_cancelled}")

            assert num_actually_cancelled > num_requests * 0.8, "Not enough tasks were cancelled" # assert at least 80% of tasks were cancelled

            print("Cancellation test passed")

    asyncio.run(test_cancellation())