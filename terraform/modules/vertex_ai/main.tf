resource "google_vertex_ai_metadata_store" "default" {
  name   = "default"
  region = var.region
}
