export interface Chunk {
  idx: number;
  text: string;
  score?: number;
}

export interface SampleDoc {
  key: string;
  title: string;
  icon: string;
  text: string;
}

export interface PipelineLog {
  icon: string;
  title: string;
  content: string;
}
