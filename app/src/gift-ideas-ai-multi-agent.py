import operator
from pydantic import BaseModel, Field
from typing import List, Literal, Annotated

import logging

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from langgraph.constants import Send
from langgraph.graph import END, MessagesState, START, StateGraph
from langgraph.types import Command
import datetime
import os
from serpapi import GoogleSearch
import json
import argparse

# Set up logging
log_formatted_str = "%(asctime)s [%(name)s] [%(levelname)s] [%(funcName)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=log_formatted_str)
logger = logging.getLogger(__name__)

### LLM

llm = ChatOpenAI(model="gpt-4o", temperature=0)

### Schema
class GiftShopper(BaseModel):
  topic: str # Gift Shopping topic
  max_ideas: int # Number of ideas to generate

class SearchQuery(BaseModel):
  shopper_type: Literal["Enthusiast", "Essentialist", "Frugalist"]
  search_query: Annotated[str, operator.add]

class Idea(BaseModel):
    name: str = Field(
        description="Name of the Idea."
    )
    description: str = Field(
        description="Description of the Idea",
    )
    shopper_type: Literal["Enthusiast", "Essentialist", "Frugalist"] = Field(
      description="Shopper Type",
    )

class IdeaList(BaseModel):
    ideas: Annotated[list[Idea], operator.add]

class WebSearchResult(BaseModel):
  title: str
  link: str
  source: str
  shopper_type: Literal["Enthusiast", "Essentialist", "Frugalist"]
  position: int
  thumbnail: str
  price: str
  tag: str
  product_link: str


class WebSearchList(BaseModel):
  search_results: Annotated[list[WebSearchResult], operator.add]


### Nodes and edges

def final_gift_recommendations(state: WebSearchList):
  logger.debug("Finalizing report")
  html_output = """
  <html>
  <head>
    <style>
      body { font-family: Arial, sans-serif; margin: 20px; }
      table { width: 100%; border-collapse: collapse; margin-bottom: 20px; }
      th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
      th { background-color: #f2f2f2; }
      h2 { color: #333; }
      .thumbnail { width: 100px; }
    </style>
  </head>
  <body>
  """

  shopper_types = ["Enthusiast", "Essentialist", "Frugalist"]
  for shopper_type in shopper_types:
    filtered_results = [result for result in state.search_results if result.shopper_type == shopper_type]
    filtered_results.sort(key=lambda x: x.position)
    top_results = filtered_results[:3]

    html_output += f"<h2>Top 3 Choices for {shopper_type}</h2>"
    html_output += """
    <table>
      <tr>
        <th>Position</th>
        <th>Thumbnail</th>
        <th>Title</th>
        <th>Link</th>
        <th>Source</th>
        <th>Price</th>
        <th>Tag</th>
      </tr>
    """
    for result in top_results:
      html_output += f"""
      <tr>
        <td>{result.position}</td>
        <td><img src="{result.thumbnail}" class="thumbnail" /></td>
        <td>{result.title}</td>
        <td><a href="{result.link}" target="_blank">{result.link}</a></td>
        <td>{result.source}</td>
        <td>{result.price}</td>
        <td>{result.tag}</td>
      </tr>
      """
    html_output += "</table>"

  html_output += """
  </body>
  </html>
  """

  with open("gift_recommendations.html", "w") as f:
    f.write(html_output)
  logger.info("HTML output saved as 'gift_recommendations.html'")

# Command[Literal["gift_shopper_the_enthusiast", "gift_shopper_the_essentialist", "gift_shopper_the_frugalist"]]
def gift_ideation_router(state: GiftShopper):
  logger.debug("Routing gift ideation")
  """ Router to select the correct gift ideation node based on the topic """
  return state

def gift_shopper(state: GiftShopper, shopper_type: str, instructions: str) -> Command[Literal["scour_the_internet"]]:
  logger.info(f"Entering {shopper_type}")
  topic = state.topic
  max_ideas = state.max_ideas
  system_message = instructions.format(topic=topic, max_ideas=max_ideas)
  structured_llm = llm.with_structured_output(IdeaList)
  llm_response = structured_llm.invoke([SystemMessage(content=system_message)] + [HumanMessage(content=f"Select {max_ideas} gift ideas.")])
  logger.info(f"LLM response: {llm_response}")
  return Send("scour_the_internet", {"ideas": llm_response.ideas})

def gift_shopper_the_enthusiast(state: GiftShopper) -> Command[Literal["scour_the_internet"]]:
  instructions = """You are a shopper who prioritizes excitement, joy, and memorable experiences, selecting gifts that bring a burst of fun and delight into the recipient’s life.

  Select 3 gift ideas on this topic: {topic}.
  """
  return gift_shopper(state, "enthusiast", instructions)

def gift_shopper_the_essentialist(state: GiftShopper) -> Command[Literal["scour_the_internet"]]:
  instructions = """You are a shopper who focuses on functional, purposeful items that seamlessly fit into daily routines, ensuring each gift is both meaningful and enduring.

  Select {max_ideas} gift ideas on this topic: {topic}.
  """
  return gift_shopper(state, "essentialist", instructions)

def gift_shopper_the_frugalist(state: GiftShopper) -> Command[Literal["scour_the_internet"]]:
  instructions = """You are a shopper who carefully curates gifts that maximize value while staying firmly within budget. A true connoisseur of cost-effective choices.

  Select {max_ideas} gift ideas on this topic: {topic}.
  """
  return gift_shopper(state, "frugalist", instructions)

def scour_the_internet(state: IdeaList) -> Command[Literal["web_search_agent"]]:
  # logger.info(f"Scouring the internet for {shopper_type}")
  """ Search query for retrieval """
  logger.info(f"*"*50)
  logger.info(f"Scouring the internet for ideas: {state}")
  logger.info(f"*"*50)
  for idea in state["ideas"]:
    search_query = f"{idea.name}: {idea.description}"
    return Send("web_search_agent", {"shopper_type": idea.shopper_type, "search_query": search_query})

def web_search_agent(state: SearchQuery) -> Command[Literal["final_gift_recommendations"]]:
  logger.info(f"-"*50)
  logger.info(f"SearchQuery state: {state}")
  logger.info(f"-"*50)
  # latest_search_query = state.search_queries[-1]
  # search_query = latest_search_query.search_query
  # shopper_type = latest_search_query.shopper_type
  search_query = state["search_query"]
  shopper_type = state["shopper_type"]

  all_results = []
  if os.getenv("USE_SIMULATE_SEARCH", "true").lower() == "true":
    logger.info("Simulating web search results")
    all_results = [
      WebSearchResult(title="Best Budget Gifts", link="http://example.com/budget-gifts", source="Example Source", shopper_type=shopper_type, position=1, thumbnail="http://example.com/thumbnail1.jpg", price="$10", tag="Budget", product_link="http://example.com/product1"),
      WebSearchResult(title="Top Essential Gifts", link="http://example.com/essential-gifts", source="Example Source", shopper_type=shopper_type, position=2, thumbnail="http://example.com/thumbnail2.jpg", price="$20", tag="Essential", product_link="http://example.com/product2"),
      WebSearchResult(title="Exciting Gift Ideas", link="http://example.com/exciting-gifts", source="Example Source", shopper_type=shopper_type, position=3, thumbnail="http://example.com/thumbnail3.jpg", price="$30", tag="Exciting", product_link="http://example.com/product3"),
      WebSearchResult(title="Affordable Tech Gadgets", link="http://example.com/tech-gadgets", source="Tech Source", shopper_type=shopper_type, position=4, thumbnail="http://example.com/thumbnail4.jpg", price="$40", tag="Tech", product_link="http://example.com/product4"),
      WebSearchResult(title="Unique Handmade Gifts", link="http://example.com/handmade-gifts", source="Craft Source", shopper_type=shopper_type, position=5, thumbnail="http://example.com/thumbnail5.jpg", price="$50", tag="Handmade", product_link="http://example.com/product5"),
      WebSearchResult(title="Eco-Friendly Gifts", link="http://example.com/eco-gifts", source="Green Source", shopper_type=shopper_type, position=6, thumbnail="http://example.com/thumbnail6.jpg", price="$60", tag="Eco", product_link="http://example.com/product6"),
      WebSearchResult(title="Luxury Gifts on a Budget", link="http://example.com/luxury-budget", source="Luxury Source", shopper_type=shopper_type, position=7, thumbnail="http://example.com/thumbnail7.jpg", price="$70", tag="Luxury", product_link="http://example.com/product7"),
      WebSearchResult(title="Practical Everyday Gifts", link="http://example.com/everyday-gifts", source="Daily Source", shopper_type=shopper_type, position=8, thumbnail="http://example.com/thumbnail8.jpg", price="$80", tag="Practical", product_link="http://example.com/product8"),
      WebSearchResult(title="Fun and Quirky Gifts", link="http://example.com/quirky-gifts", source="Fun Source", shopper_type=shopper_type, position=9, thumbnail="http://example.com/thumbnail9.jpg", price="$90", tag="Quirky", product_link="http://example.com/product9"),
      WebSearchResult(title="Top Gifts for Enthusiasts", link="http://example.com/enthusiast-gifts", source="Enthusiast Source", shopper_type=shopper_type, position=10, thumbnail="http://example.com/thumbnail10.jpg", price="$100", tag="Enthusiast", product_link="http://example.com/product10")
    ]
  else:
    serpapi_api_key = os.getenv("SERPAPI_API_KEY", "")
    params = {
      "q": search_query,
      "api_key": serpapi_api_key,
      "engine": "google_shopping",
      "google_domain": "google.com",
      "direct_link": "true",
      "gl": "us",
      "hl": "en",
      "num": "5"
    }
    search = GoogleSearch(params)
    all_results = []
    results = search.get_dict()
    logger.info("Search results from SerpAPI:")
    logger.info(json.dumps(results, indent=2))
    for result in results.get('shopping_results', []):
      position = result.get('position', 1)
      source = result.get('source', 'Unknown Source')
      title = result.get('title', 'No Title')
      link = result.get('link', 'No Link')
      thumbnail = result.get('thumbnail', 'No Thumbnail')
      price = result.get('price', 'No Price')
      tag = result.get('tag', 'No Tag')
      product_link = result.get('product_link', 'No Product Link')
      all_results.append(
        WebSearchResult(
          title=title,
          link=link,
          source=source,
          shopper_type=shopper_type,
          position=position,
          thumbnail=thumbnail,
          price=price,
          tag=tag,
          product_link=product_link
        )
      )
  logger.info(f"All search results: {all_results}")

  return Command(
    goto="final_gift_recommendations",
    update={"search_results": all_results}
  )

# Add nodes and edges
builder = StateGraph(GiftShopper)
builder.add_node("gift_ideation_router", gift_ideation_router)
builder.add_node("gift_shopper_the_enthusiast", gift_shopper_the_enthusiast)
builder.add_node("gift_shopper_the_essentialist", gift_shopper_the_essentialist)
builder.add_node("gift_shopper_the_frugalist", gift_shopper_the_frugalist)
builder.add_node("scour_the_internet", scour_the_internet)
builder.add_node("web_search_agent", web_search_agent)
builder.add_node("final_gift_recommendations", final_gift_recommendations)

# Logic
builder.add_edge(START, "gift_ideation_router")
builder.add_edge("gift_ideation_router", "gift_shopper_the_enthusiast")
builder.add_edge("gift_ideation_router", "gift_shopper_the_essentialist")
builder.add_edge("gift_ideation_router", "gift_shopper_the_frugalist")
builder.add_edge("final_gift_recommendations", END)

# Compile
graph = builder.compile()

def create_agent_graph_image():
  graph_image = graph.get_graph(xray=1).draw_mermaid_png()
  timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
  filename = f"agent_{timestamp}.png"
  with open(filename, "wb") as f:
    f.write(graph_image)
  logger.info(f"Graph image saved as '{filename}'. Open this file to view the graph.")

if __name__ == "__main__":

  parser = argparse.ArgumentParser(description="Gift Ideas AI Multi-Agent System")
  parser.add_argument("--topic", type=str, required=True, help="Gift Shopping topic")
  parser.add_argument("--max_ideas", type=int, required=True, help="Number of ideas to generate")
  parser.add_argument("--generate-graph", action="store_true", help="Generate and save the state graph as an image")

  args = parser.parse_args()

  if args.generate_graph:
    create_agent_graph_image()

  graph.invoke({"topic": args.topic, "max_ideas": args.max_ideas})

