# Requirements Document

## Introduction

The Product Recommendation Engine is a feature for the MUG VN - MDB Hackathon that delivers personalized product recommendations to users during product search. The system extracts product data from multiple PostgreSQL sources (Odoo ERP and VTM system), uploads it to MongoDB, and generates vector embeddings stored alongside product data. When a user performs a search, the engine combines the query with the user's behavioral context (search history, clicks, purchases) to retrieve the most semantically relevant products using MongoDB's vector search capabilities.

## Glossary

- **Recommendation_Engine**: The core system responsible for generating and serving personalized product recommendations.
- **Data_Pipeline**: The ETL process that extracts product data from PostgreSQL sources, transforms it, and loads it into MongoDB.
- **Embedding_Service**: The component responsible for generating vector embeddings from product data and user queries.
- **Vector_Store**: The MongoDB collection that stores product data alongside their vector embeddings.
- **Behavior_Tracker**: The component that captures and stores user behavioral signals (searches, clicks, purchases).
- **User_Context**: The aggregated behavioral profile of a user, derived from search history, click events, and purchase history.
- **Semantic_Search**: A search method that uses vector similarity to find products relevant to the meaning of a query, beyond keyword matching.
- **Odoo_ERP**: The Odoo-based ERP PostgreSQL database, one of the product data sources.
- **VTM_System**: The VTM PostgreSQL database, one of the product data sources.
- **Product**: A catalog item with attributes such as name, description, category, price, and metadata sourced from Odoo_ERP or VTM_System.
- **Embedding**: A high-dimensional numerical vector representation of a product or query, used for semantic similarity computation.
- **Similarity_Score**: A numerical value representing the semantic closeness between a query embedding and a product embedding.

---

## Requirements

### Requirement 1: Data Extraction from PostgreSQL Sources

**User Story:** As a data engineer, I want to extract product data from Odoo ERP and VTM PostgreSQL databases, so that the recommendation engine has access to a unified and up-to-date product catalog.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL connect to the Odoo_ERP PostgreSQL database and extract product records including name, description, category, price, and availability.
2. THE Data_Pipeline SHALL connect to the VTM_System PostgreSQL database and extract product records including name, description, category, price, and availability.
3. WHEN a product record is extracted, THE Data_Pipeline SHALL assign a unique identifier that preserves the source system and original record ID.
4. WHEN a product record already exists in the Vector_Store with the same source identifier, THE Data_Pipeline SHALL update the existing record rather than create a duplicate.
5. IF a PostgreSQL connection fails, THEN THE Data_Pipeline SHALL log the error with the source system name, timestamp, and failure reason, and SHALL continue processing remaining sources.
6. THE Data_Pipeline SHALL support both full extraction (all products) and incremental extraction (products modified since the last successful run).

---

### Requirement 2: Data Loading into MongoDB

**User Story:** As a data engineer, I want extracted product data to be loaded into MongoDB, so that the system has a centralized store for products and their embeddings.

#### Acceptance Criteria

1. WHEN product records are extracted from a PostgreSQL source, THE Data_Pipeline SHALL upload them to the designated MongoDB collection in the Vector_Store.
2. THE Data_Pipeline SHALL preserve all extracted product fields when writing to MongoDB.
3. WHEN a batch upload to MongoDB fails, THE Data_Pipeline SHALL retry the upload up to 3 times before logging the failure and skipping the batch.
4. THE Data_Pipeline SHALL record the timestamp of each successful load operation per source system.

---

### Requirement 3: Embedding Generation and Storage

**User Story:** As a data engineer, I want vector embeddings generated for each product and stored in MongoDB, so that semantic similarity search can be performed at query time.

#### Acceptance Criteria

1. WHEN a product record is loaded into MongoDB, THE Embedding_Service SHALL generate a vector embedding from the product's name, description, and category fields.
2. THE Embedding_Service SHALL store the generated embedding in the same MongoDB document as the product data.
3. WHEN a product record is updated in MongoDB, THE Embedding_Service SHALL regenerate and overwrite the existing embedding for that product.
4. THE Embedding_Service SHALL use a consistent embedding model across all products so that embeddings are comparable by vector similarity.
5. IF embedding generation fails for a product, THEN THE Embedding_Service SHALL log the product identifier and error reason, and SHALL NOT store a partial or empty embedding for that product.
6. THE Vector_Store SHALL maintain a MongoDB Atlas Vector Search index on the embedding field to enable efficient similarity queries.

---

### Requirement 4: User Behavior Tracking

**User Story:** As a product manager, I want the system to capture user behavioral signals, so that recommendations can be personalized based on each user's interests and intent.

#### Acceptance Criteria

1. WHEN a user submits a search query, THE Behavior_Tracker SHALL record the query text, user identifier, and timestamp.
2. WHEN a user clicks on a product in search results, THE Behavior_Tracker SHALL record the product identifier, user identifier, and timestamp.
3. WHEN a user completes a purchase, THE Behavior_Tracker SHALL record the purchased product identifiers, user identifier, and timestamp.
4. THE Behavior_Tracker SHALL store all behavioral events in MongoDB associated with the user identifier.
5. WHILE a user session is active, THE Behavior_Tracker SHALL capture behavioral events in real time with a latency of no more than 500ms from event occurrence to storage.
6. IF a user identifier is not available (anonymous user), THEN THE Behavior_Tracker SHALL associate behavioral events with a session identifier instead.

---

### Requirement 5: User Context Aggregation

**User Story:** As a data scientist, I want the system to build a behavioral context profile for each user, so that the recommendation engine can incorporate personalization signals into search results.

#### Acceptance Criteria

1. WHEN a recommendation request is received, THE Recommendation_Engine SHALL retrieve the user's most recent 50 behavioral events from MongoDB.
2. THE Recommendation_Engine SHALL weight purchase events more heavily than click events, and click events more heavily than search events, when constructing the User_Context.
3. THE Recommendation_Engine SHALL derive a context embedding by combining the current search query embedding with the User_Context signals.
4. IF no behavioral history exists for a user, THEN THE Recommendation_Engine SHALL use only the search query embedding as the context for retrieval.

---

### Requirement 6: Semantic Product Search and Recommendation

**User Story:** As a user, I want to receive personalized product recommendations when I search, so that I can quickly find products most relevant to my needs and interests.

#### Acceptance Criteria

1. WHEN a user submits a search query, THE Recommendation_Engine SHALL generate an embedding for the query using the same model used for product embeddings.
2. WHEN a context embedding is available, THE Recommendation_Engine SHALL perform a vector similarity search against the Vector_Store using the context embedding.
3. THE Recommendation_Engine SHALL return the top-K most similar products, where K is configurable and defaults to 10.
4. THE Recommendation_Engine SHALL include the Similarity_Score for each recommended product in the response.
5. THE Recommendation_Engine SHALL return recommendation results within 2 seconds of receiving a search request under normal operating conditions.
6. WHERE a minimum similarity threshold is configured, THE Recommendation_Engine SHALL exclude products with a Similarity_Score below that threshold from the results.
7. THE Recommendation_Engine SHALL return results as a ranked list ordered by Similarity_Score in descending order.

---

### Requirement 7: API Interface

**User Story:** As a frontend developer, I want a well-defined API to query the recommendation engine, so that I can integrate product recommendations into the search UI.

#### Acceptance Criteria

1. THE Recommendation_Engine SHALL expose an HTTP API endpoint that accepts a search query and a user identifier as input parameters.
2. WHEN a valid request is received, THE Recommendation_Engine SHALL respond with a JSON payload containing the ranked list of recommended products and their Similarity_Scores.
3. IF a request is missing required parameters, THEN THE Recommendation_Engine SHALL return an HTTP 400 response with a descriptive error message.
4. IF an internal error occurs during recommendation generation, THEN THE Recommendation_Engine SHALL return an HTTP 500 response with an error code and SHALL log the full error details server-side.
5. THE Recommendation_Engine SHALL support pagination of results through `page` and `page_size` query parameters.

---

### Requirement 8: Data Pipeline Orchestration

**User Story:** As a data engineer, I want the data pipeline to run on a schedule and be triggerable on demand, so that the product catalog and embeddings stay current without manual intervention.

#### Acceptance Criteria

1. THE Data_Pipeline SHALL support scheduled execution at a configurable interval (default: every 24 hours).
2. THE Data_Pipeline SHALL support on-demand manual triggering via an administrative API or CLI command.
3. WHEN a pipeline run completes, THE Data_Pipeline SHALL record the run status (success or failure), start time, end time, and number of records processed per source.
4. IF a pipeline run is already in progress, THEN THE Data_Pipeline SHALL reject a new trigger request and return a status message indicating a run is in progress.
