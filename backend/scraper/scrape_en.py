import asyncio
import os
import csv
import re
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.deep_crawling.filters import (
    FilterChain,
    DomainFilter,
    URLPatternFilter
)

# Function to create a valid, safe filename from a URL
def safe_filename(url):
    # Remove the protocol (http/https) and 'www.'
    filename = re.sub(r'https?://(www\.)?', '', url)
    # Replace slashes and other non-alphanumeric characters with underscores
    # Keep only letters, numbers, hyphens, and dots, replace everything else with '_'
    filename = re.sub(r'[^a-zA-Z0-9\-\.]', '_', filename)
    # Trim trailing underscores that might result from trailing slashes
    filename = filename.strip('_')
    # Prepend 'index' if the filename is empty (e.g., just the root domain)
    if not filename:
        return "index.md"
    # Ensure it ends with .md
    if not filename.endswith('.md'):
        filename += '.md'
    return filename

async def main():
    # Define the exact hostname you want to stick to
    TARGET_DOMAIN = "www.fhs.hr"
    TARGET_URL = f"https://{TARGET_DOMAIN}"

    # 1. Define the Domain Filter (Sticking to www.fhs.hr)
    domain_filter = DomainFilter(
        allowed_domains=[TARGET_DOMAIN]
    )

    # 2. NEW FILTER: Strictly include only paths starting with /en/
    english_path_filter = URLPatternFilter(
    # We require the full domain followed by /en/ to prevent veering off
    patterns=[
    f"https://{TARGET_DOMAIN}/en/*",
    f"https://{TARGET_DOMAIN}/staff/*"
    ])
    # reverse is False by default, meaning it acts as an inclusion filter 
    
    # 2. Define the URL Exclusion Filter
    url_exclusion_filter = URLPatternFilter(
        patterns=[
            re.compile(r"\?@=21"),  # ← literal ?@=21 as regex
            "*news*", 
            "*obavijesti*",
            "*_download/repository*", 
            "*/images/*",
            "*\\?@=21*",  # ← literal ? before @=21,
            "*21*",  # ← literal ? before @=21,
            "*saml-login*",
            "*login*",
            "*lid*",
            "*lid=*",
            "*2024*",
            "*2023*",
            "*2022*",
            "*2021*",
            "*2020*",
            "*=*"
        ],
        reverse=True
    )

    # 3. Create a Filter Chain containing BOTH filters
    boundary_filter_chain = FilterChain([
        domain_filter,     # 1. Must be on TARGET_DOMAIN
        english_path_filter,  # NEW: Must be under /en/ path
        url_exclusion_filter  # 2. Must NOT contain the excluded patterns
    ])

    # --- Configuration is updated to include the filter_chain ---
    config = CrawlerRunConfig(
        target_elements=[
        ".col-xl-8",
        ".cms_module portlet_content"
        ".cms_module portlet_predmet_info"      
        ],
        deep_crawl_strategy=BFSDeepCrawlStrategy(
            max_depth=10, 
            include_external=False,
            filter_chain=boundary_filter_chain 
        ),
        scraping_strategy=LXMLWebScrapingStrategy(),
        verbose=True,
        # Tag exclusions
        excluded_tags=['form', 'header', 'footer'],
        semaphore_count=5,
        stream=True, # <-- KEEPING STREAM=TRUE
        cache_mode=CacheMode.BYPASS,
        # Dynamic Content Handling
        wait_until="networkidle",

        # Link filtering
        exclude_external_links=True,  
        exclude_social_media_links=True,
        # Block entire domains
        exclude_domains=["adtrackers.com", "spammynews.org"],  
        exclude_social_media_domains=["facebook.com", "twitter.com"],

        # Media filtering
        exclude_external_images=True
        )

    output_dir = "crawled_output_en"
    os.makedirs(output_dir, exist_ok=True) # Create the output directory if it doesn't exist

    # File to log processed URLs (CSV)
    processed_csv = "processed_urls_en.csv"

    # Initialize CSV file with headers if it doesn't exist yet
    if not os.path.exists(processed_csv):
        with open(processed_csv, "w", encoding="utf-8", newline="") as cf:
            writer = csv.writer(cf)
            writer.writerow(["URL", "Saved File", "Depth"])

    crawled_count = 0
    async with AsyncWebCrawler() as crawler:
        # CRITICAL FIX: Use async for to consume the stream
        async for result in await crawler.arun(TARGET_URL, config=config):
            filename = safe_filename(result.url)
            filepath = os.path.join(output_dir, filename)

            try:
                if result.markdown:
                    # --- NEW LOGIC: Prepend the link header to the markdown content ---
                    link_header = f"[Article Link]({result.url})\n\n"
                    final_markdown = link_header + result.markdown
                    
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(final_markdown)
                    # --- END NEW LOGIC ---

                    # Append processed URL info to CSV
                    with open(processed_csv, "a", encoding="utf-8", newline="") as cf:
                        writer = csv.writer(cf)
                        writer.writerow([result.url, filepath, result.metadata.get("depth", 0)])

                    crawled_count += 1

                else:
                    print(f"Skipping URL {result.url}: No markdown content extracted.")

            except IOError as e:
                print(f" ERROR saving file {filepath} for URL: {result.url}. Error: {e}")

    # Print the final count after the stream is fully consumed
    print(f"Crawled {crawled_count} pages in total")

if __name__ == "__main__":
    asyncio.run(main())