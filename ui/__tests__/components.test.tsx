import { render, screen, fireEvent } from '@testing-library/react';
import SearchBar from '@/components/SearchBar';
import RepoCard from '@/components/RepoCard';
import SearchResults from '@/components/SearchResults';
import { RepoResult } from '@/lib/api';

describe('SearchBar', () => {
  it('renders search input and button', () => {
    const mockOnSearch = jest.fn();
    render(<SearchBar onSearch={mockOnSearch} />);
    
    expect(screen.getByPlaceholderText('Search repositories...')).toBeInTheDocument();
    expect(screen.getByText('Search')).toBeInTheDocument();
  });

  it('calls onSearch when form is submitted', () => {
    const mockOnSearch = jest.fn();
    render(<SearchBar onSearch={mockOnSearch} />);
    
    const input = screen.getByPlaceholderText('Search repositories...');
    const button = screen.getByText('Search');
    
    fireEvent.change(input, { target: { value: 'test query' } });
    fireEvent.click(button);
    
    expect(mockOnSearch).toHaveBeenCalledWith('test query');
  });

  it('disables button when loading', () => {
    const mockOnSearch = jest.fn();
    render(<SearchBar onSearch={mockOnSearch} isLoading={true} />);
    
    const button = screen.getByText('Searching...');
    expect(button).toBeDisabled();
  });
});

describe('RepoCard', () => {
  const mockRepo: RepoResult = {
    repo_name: 'test-repo',
    full_name: 'owner/test-repo',
    description: 'Test description',
    summary: 'Test summary',
    tags: ['python', 'ml'],
    language: 'Python',
    stars: 100,
    url: 'https://github.com/owner/test-repo',
    score: 0.95,
  };

  it('renders repository information', () => {
    render(<RepoCard repo={mockRepo} />);
    
    expect(screen.getByText('owner/test-repo')).toBeInTheDocument();
    expect(screen.getByText('Test description')).toBeInTheDocument();
    expect(screen.getByText('Test summary')).toBeInTheDocument();
    expect(screen.getByText('Python')).toBeInTheDocument();
    expect(screen.getByText('⭐ 100')).toBeInTheDocument();
  });

  it('renders tags', () => {
    render(<RepoCard repo={mockRepo} />);
    
    expect(screen.getByText('python')).toBeInTheDocument();
    expect(screen.getByText('ml')).toBeInTheDocument();
  });
});

describe('SearchResults', () => {
  const mockResults: RepoResult[] = [
    {
      repo_name: 'repo1',
      full_name: 'owner/repo1',
      description: 'Repo 1',
      summary: null,
      tags: [],
      language: 'Python',
      stars: 50,
      url: 'https://github.com/owner/repo1',
      score: 0.9,
    },
  ];

  it('renders results count', () => {
    render(<SearchResults results={mockResults} query="test" />);
    
    expect(screen.getByText('Found 1 repositories')).toBeInTheDocument();
  });

  it('shows no results message when empty', () => {
    render(<SearchResults results={[]} query="test" />);
    
    expect(screen.getByText(/No results found for "test"/)).toBeInTheDocument();
  });
});
