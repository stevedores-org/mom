package main

import (
	"context"
	"fmt"
	"io"
	"log"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"

	// These imports assume you have generated Go code from memory.proto
	// command: protoc --go_out=. --go-grpc_out=. protos/memory.proto
	pb "github.com/lornu-ai/mom/proto/memory"
)

func main() {
	// 1. Establish connection
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	conn, err := grpc.DialContext(ctx, "localhost:50051", 
		grpc.WithTransportCredentials(insecure.NewCredentials()), 
		grpc.WithBlock(),
	)
	if err != nil {
		log.Fatalf("failed to connect to MOM gRPC server: %v", err)
	}
	defer conn.Close()
	fmt.Println("✅ Connected to MOM gRPC server on localhost:50051")

	client := pb.NewMemoryStoreServiceClient(conn)

	// Define scope
	scope := &pb.ScopeKey{
		TenantId:    "example-tenant",
		WorkspaceId: stringPtr("workspace-1"),
		AgentId:     stringPtr("agent-go"),
	}

	// 2. Write Memory
	writeReq := &pb.MemoryItem{
		Id:          "doc-go-123",
		Scope:       scope,
		Kind:        pb.MemoryKind_MEMORY_KIND_FACT,
		CreatedAtMs: time.Now().UnixMilli(),
		Content: &pb.Content{
			ContentType: &pb.Content_Text{
				Text: "Go client successfully connected and wrote a fact.",
			},
		},
		Tags:       []string{"go", "client", "test"},
		Importance: 0.85,
		Confidence: 0.99,
		Source:     "go-agent",
	}

	fmt.Println("Writing memory doc-go-123...")
	writeResp, err := client.Write(context.Background(), writeReq)
	if err != nil {
		log.Fatalf("Failed to write memory: %v", err)
	}
	fmt.Printf("✅ Written successfully. ID: %s\n", writeResp.Id)

	// 3. Get Memory
	fmt.Println("Retrieving memory doc-go-123...")
	getResp, err := client.Get(context.Background(), &pb.MemoryId{
		Id:    "doc-go-123",
		Scope: scope,
	})
	if err != nil {
		log.Fatalf("Failed to get memory: %v", err)
	}
	fmt.Printf("✅ Retrieved memory: %v\n", getResp)

	// 4. Query Memories (Streaming)
	fmt.Println("Querying memories with tag 'go'...")
	queryStream, err := client.Query(context.Background(), &pb.Query{
		Scope:   scope,
		TagsAny: []string{"go"},
		Limit:   5,
	})
	if err != nil {
		log.Fatalf("Failed to query memories: %v", err)
	}

	for {
		item, err := queryStream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			log.Fatalf("Error reading stream: %v", err)
		}
		fmt.Printf("✅ Query result: Score: %f, Item ID: %s, Text: %s\n", 
			item.Score, item.Item.Id, item.Item.GetContent().GetText())
	}

	// 5. Delete Memory
	fmt.Println("Deleting memory doc-go-123...")
	_, err = client.Delete(context.Background(), &pb.MemoryId{
		Id:    "doc-go-123",
		Scope: scope,
	})
	if err != nil {
		log.Fatalf("Failed to delete memory: %v", err)
	}
	fmt.Println("✅ Deleted memory successfully")
}

func stringPtr(s string) *string {
	return &s
}
